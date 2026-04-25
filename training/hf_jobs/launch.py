"""Launch SFT or PPO training on HuggingFace Jobs.

Usage:
    # Quick smoke (4B model, default H200, dry-run prints the spec without
    # firing — works without HF Pro):
    python -m training.hf_jobs.launch sft \\
        --base-model Qwen/Qwen3-4B-Instruct-2507 \\
        --sft-data artifacts/sft/sft-warmstart.jsonl \\
        --output-repo adityabhaskara/dronecaptureops-sft-qwen3-4b \\
        --dry-run

    # Real fire (requires HF Pro):
    python -m training.hf_jobs.launch sft \\
        --base-model Qwen/Qwen3-32B \\
        --sft-data artifacts/sft/sft-warmstart.jsonl \\
        --output-repo adityabhaskara/dronecaptureops-sft-qwen3-32b \\
        --hardware h200 \\
        --follow

    # PPO from a previously-trained SFT adapter (when the PPO trainer
    # exists; the launcher path is ready):
    python -m training.hf_jobs.launch ppo \\
        --base-model adityabhaskara/dronecaptureops-sft-qwen3-32b \\
        --output-repo adityabhaskara/dronecaptureops-ppo-qwen3-32b \\
        --hardware h200 \\
        --follow

The launcher always:
  1. Verifies HF account access (whoami + Pro check).
  2. Pushes the current repo (as a private dataset) and the SFT JSONL.
  3. Builds a JobSpec for the requested job type.
  4. Submits via huggingface_hub.run_job().
  5. Optionally tails logs (`--follow`).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from training.hf_jobs.job_specs import (
    DEFAULT_HARDWARE_BY_JOB,
    DEFAULT_IMAGE,
    DEFAULT_TIMEOUT_BY_JOB,
    JobSpec,
    JobType,
    build_job_spec,
)
from training.hf_jobs.push_artifacts import (
    git_describe,
    push_jsonl_dataset,
    push_repo_tarball,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger("dronecaptureops.hf_jobs.launch")


# ---------------------------------------------------------------------------
# Account & access verification
# ---------------------------------------------------------------------------


class AccountError(SystemExit):
    """Raised when the user's HF account can't run jobs."""


def verify_hf_access(token: str) -> dict:
    """Confirm whoami works and that the account can submit jobs.

    HF Jobs is gated behind Pro / Enterprise — without it list_jobs
    returns 403. We surface a clear error early instead of letting
    run_job() fail mid-flight after we've already pushed artefacts.
    """

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        whoami = api.whoami()
    except Exception as exc:  # noqa: BLE001
        raise AccountError(
            f"HF whoami() failed — check your HF_TOKEN / HF_AUTH_TOKEN. Underlying error: {exc}"
        ) from exc

    LOG.info("authenticated as %s (type=%s, pro=%s)", whoami.get("name"), whoami.get("type"), whoami.get("isPro"))

    try:
        api.list_jobs()
    except Exception as exc:  # noqa: BLE001
        raise AccountError(
            "HF Jobs is not available on this account.\n"
            f"  Underlying error: {exc}\n"
            "  HF Jobs requires a Pro subscription ($9/mo) or HF Enterprise org membership.\n"
            "  Upgrade at https://huggingface.co/settings/billing then retry."
        ) from exc

    return whoami


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------


def derive_repo_dataset_id(*, namespace: str, base_model: str, job_type: str) -> str:
    """Default name for the per-launch repo snapshot dataset."""

    safe = base_model.replace("/", "__")
    return f"{namespace}/dronecaptureops-{job_type}-bundle-{safe}"


def derive_data_dataset_id(*, namespace: str, base_model: str) -> str:
    safe = base_model.replace("/", "__")
    return f"{namespace}/dronecaptureops-sft-data-{safe}"


def submit(spec: JobSpec, *, token: str, dry_run: bool = False, namespace: str | None = None) -> str | None:
    """Fire the job (or print the resolved spec, when dry-run).

    Returns the job id when submitted, None when dry-run.
    """

    if dry_run:
        LOG.info("DRY RUN — would call huggingface_hub.run_job() with:")
        kwargs = spec.to_run_job_kwargs()
        kwargs["secrets"] = {key: "<redacted>" for key in kwargs.get("secrets", {})}
        for key, value in kwargs.items():
            LOG.info("  %s = %r", key, value)
        return None

    from huggingface_hub import run_job

    kwargs = spec.to_run_job_kwargs()
    kwargs["token"] = token
    if namespace:
        kwargs["namespace"] = namespace
    info = run_job(**kwargs)
    job_id = getattr(info, "id", None) or getattr(info, "job_id", None) or str(info)
    LOG.info("submitted job_id=%s", job_id)
    return job_id


def follow_logs(*, job_id: str, token: str, namespace: str | None = None) -> None:
    """Stream job logs to stdout until the job finishes."""

    from huggingface_hub import fetch_job_logs, inspect_job

    LOG.info("tailing logs for job %s (Ctrl+C to detach; the job keeps running)", job_id)
    for line in fetch_job_logs(job_id=job_id, token=token, namespace=namespace, follow=True):
        sys.stdout.write(line if line.endswith("\n") else line + "\n")
        sys.stdout.flush()

    final = inspect_job(job_id=job_id, token=token, namespace=namespace)
    LOG.info("final status: %s", getattr(final, "status", final))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch DroneCaptureOps SFT or PPO on HF Jobs.")
    parser.add_argument("job_type", choices=["sft", "ppo"], help="Which trainer to run.")
    parser.add_argument("--base-model", required=True, help="HF model ID to train (or pre-SFT'd checkpoint for PPO).")
    parser.add_argument("--output-repo", required=True, help="HF model repo to receive the trained adapter.")
    parser.add_argument("--sft-data", type=Path, default=Path("artifacts/sft/sft-warmstart.jsonl"),
                        help="Local path to the SFT JSONL. SFT jobs require this.")
    parser.add_argument("--config", default=None,
                        help="Repo-relative path to the YAML config the trainer should consume. "
                             "Defaults to training/configs/sft_train_default.yaml or ppo_train_default.yaml.")
    parser.add_argument("--hardware", default=None, help="HF Jobs flavor (e.g. h200, h200x2, a100-large).")
    parser.add_argument("--timeout", default=None, help="Job timeout (e.g. 4h, 12h).")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Container image to run inside.")
    parser.add_argument("--namespace", default=None,
                        help="Owner namespace for the job (defaults to the token's whoami).")
    parser.add_argument("--repo-dataset-id", default=None,
                        help="Override the dataset repo where the code tarball is uploaded.")
    parser.add_argument("--data-dataset-id", default=None,
                        help="Override the dataset repo where the SFT JSONL is uploaded.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the resolved JobSpec without uploading or firing.")
    parser.add_argument("--skip-push-repo", action="store_true",
                        help="Skip uploading the repo tarball (assumes it's already in --repo-dataset-id).")
    parser.add_argument("--skip-push-data", action="store_true",
                        help="Skip uploading the SFT JSONL (assumes it's already in --data-dataset-id).")
    parser.add_argument("--follow", action="store_true",
                        help="Tail the job logs after submission.")
    parser.add_argument("--extra-arg", action="append", default=[],
                        help="Extra positional arg to append to the in-job entrypoint command (repeatable).")
    return parser.parse_args()


def _resolve_token() -> str:
    for var in ("HF_TOKEN", "HF_AUTH_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(var)
        if value:
            return value
    raise AccountError("No HF token found in env (HF_TOKEN / HF_AUTH_TOKEN / HUGGINGFACE_TOKEN).")


def _resolve_default_config(job_type: JobType) -> str:
    if job_type == "sft":
        return "training/configs/sft_train_default.yaml"
    return "training/configs/ppo_train_default.yaml"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args()

    token = _resolve_token()
    whoami = verify_hf_access(token) if not args.dry_run else None
    namespace = args.namespace or (whoami.get("name") if whoami else "<your-namespace>")

    repo_dataset_id = args.repo_dataset_id or derive_repo_dataset_id(
        namespace=namespace, base_model=args.base_model, job_type=args.job_type
    )
    data_dataset_id = args.data_dataset_id or derive_data_dataset_id(
        namespace=namespace, base_model=args.base_model
    )
    config_path = args.config or _resolve_default_config(args.job_type)

    rev = git_describe(REPO_ROOT)
    extra_env: dict[str, str] = {}
    if rev:
        extra_env["DRONECAPTUREOPS_GIT_REV"] = rev

    # 1. Push the repo + SFT data (skipped when --dry-run unless caller asks).
    if not args.dry_run:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        if not args.skip_push_repo:
            LOG.info("pushing repo snapshot to %s", repo_dataset_id)
            push_repo_tarball(api, repo_root=REPO_ROOT, dataset_repo_id=repo_dataset_id)
        if args.job_type == "sft" and not args.skip_push_data:
            if not args.sft_data.exists():
                raise SystemExit(
                    f"SFT data not found at {args.sft_data}. "
                    "Run training/generate_sft_data.py first or pass --sft-data."
                )
            LOG.info("pushing SFT data to %s", data_dataset_id)
            push_jsonl_dataset(api, jsonl_path=args.sft_data, dataset_repo_id=data_dataset_id)

    # 2. Build the job spec.
    spec = build_job_spec(
        job_type=args.job_type,
        repo_dataset_id=repo_dataset_id,
        data_dataset_id=data_dataset_id,
        output_repo_id=args.output_repo,
        base_model=args.base_model,
        config_path_in_repo=config_path,
        hf_token=token,
        hardware=args.hardware,
        timeout=args.timeout,
        image=args.image,
        extra_env=extra_env,
        extra_args=args.extra_arg,
    )

    # 3. Submit (or print).
    job_id = submit(spec, token=token, dry_run=args.dry_run, namespace=args.namespace)

    if job_id and args.follow:
        # Brief pause so the job has a chance to enter RUNNING before we tail.
        time.sleep(2)
        follow_logs(job_id=job_id, token=token, namespace=args.namespace)

    return 0


if __name__ == "__main__":
    sys.exit(main())
