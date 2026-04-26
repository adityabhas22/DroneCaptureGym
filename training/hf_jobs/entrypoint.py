"""Entrypoint executed inside the HF Jobs container.

The launcher submits this module as the job's command. Inside the
container we:

  1. Read the env vars the launcher set (repo dataset, data dataset,
     output repo, base model, config path).
  2. Download + extract the repo snapshot.
  3. Install repo deps (best-effort: prefer pre-baked image deps).
  4. Download the SFT JSONL into the extracted repo.
  5. Run the trainer (sft_warmstart.py or train_ppo.py) with the right
     base model + config + dataset path.
  6. Push the trained adapter, training logs, and any audit JSONL
     back to the output model repo.

Everything assumes HF_TOKEN is available in the env (the launcher passes
it through the job secrets).
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


WORK_ROOT = Path("/tmp/dronecaptureops_job")
LOG = logging.getLogger("dronecaptureops.hf_jobs.entrypoint")


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"required env var {name} not set; the launcher must populate it."
        )
    return value


def _hf_token() -> str:
    for name in ("HF_TOKEN", "HF_AUTH_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(name)
        if value:
            # Make sure it's visible under the canonical name for child processes.
            os.environ.setdefault("HF_TOKEN", value)
            return value
    raise SystemExit("no HF token in env; the launcher must pass HF_TOKEN as a secret.")


def _configure_gpu_runtime_env() -> None:
    """Set CUDA/vLLM startup guards before child training processes import torch."""

    os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def _wait_for_cuda_ready(*, attempts: int | None = None, sleep_secs: int | None = None) -> None:
    """Wait for HF GPU hosts, especially H200, to finish CUDA fabric init.

    The probe runs in a subprocess so a failed CUDA runtime initialization
    cannot poison the long-lived entrypoint process. Once it succeeds, the
    actual trainer still starts in a fresh child process.

    H200 nodes hit a known HF infra bug (huggingface_hub#4128) where the
    NVIDIA Fabric Manager stays in "In Progress" state indefinitely; on
    those nodes no amount of waiting will help. We give a generous budget
    (~12 minutes by default) so healthy nodes finish init reliably while
    bad nodes still bail before the expensive download/load phase.
    """

    if attempts is None:
        attempts = int(os.environ.get("DRONECAPTUREOPS_CUDA_PROBE_ATTEMPTS", "30"))
    if sleep_secs is None:
        sleep_secs = int(os.environ.get("DRONECAPTUREOPS_CUDA_PROBE_SLEEP", "25"))

    probe = (
        "import torch, subprocess; "
        "torch.cuda.init(); "
        "n=torch.cuda.device_count(); "
        "assert n > 0, 'no cuda devices'; "
        "x=torch.empty(1, device='cuda'); "
        "print({'cuda_devices': n, 'device': torch.cuda.get_device_name(0)})"
    )
    env = os.environ.copy()
    for attempt in range(1, attempts + 1):
        result = subprocess.run(
            [sys.executable, "-c", probe],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            LOG.info("CUDA readiness probe passed: %s", result.stdout.strip())
            return
        LOG.warning(
            "CUDA readiness probe failed (%d/%d): %s%s",
            attempt,
            attempts,
            result.stderr.strip(),
            f" | stdout={result.stdout.strip()}" if result.stdout.strip() else "",
        )
        # On H200 the Fabric Manager bug never resolves; surface nvidia-smi state
        # every few attempts so we can distinguish "transient init delay" from
        # "node is permanently broken" without manual log inspection.
        if attempt % 5 == 0:
            try:
                smi = subprocess.run(
                    ["nvidia-smi", "-q", "-d", "FABRIC"],
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=15,
                )
                LOG.warning("nvidia-smi fabric snapshot (attempt %d): %s", attempt, smi.stdout.strip()[:1000])
            except Exception as exc:  # noqa: BLE001
                LOG.warning("nvidia-smi snapshot failed: %s", exc)
        if attempt < attempts:
            time.sleep(sleep_secs)
    raise SystemExit(
        "CUDA did not become ready before training; refusing to start expensive run. "
        "If this is an H200 node, the host likely hit huggingface_hub#4128 (Fabric "
        "Manager stuck). Re-launch — different node assignment usually recovers."
    )


# ---------------------------------------------------------------------------
# Step 1-2: pull repo + data
# ---------------------------------------------------------------------------


def _download_repo(token: str, dataset_repo_id: str) -> Path:
    from huggingface_hub import HfApi

    from training.hf_jobs.push_artifacts import download_repo_tarball

    LOG.info("downloading repo from %s", dataset_repo_id)
    api = HfApi(token=token)
    extracted = download_repo_tarball(
        api,
        dataset_repo_id=dataset_repo_id,
        target_dir=WORK_ROOT / "repo",
    )
    LOG.info("repo extracted to %s", extracted)
    return extracted


def _download_dataset(token: str, dataset_repo_id: str, target_dir: Path) -> Path:
    from huggingface_hub import HfApi

    target_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=token)
    LOG.info("downloading dataset files from %s", dataset_repo_id)
    files = api.list_repo_files(dataset_repo_id, repo_type="dataset")
    jsonl_files = [f for f in files if f.endswith(".jsonl")]
    if not jsonl_files:
        raise SystemExit(f"no .jsonl files in dataset {dataset_repo_id}")
    primary = jsonl_files[0]
    local = api.hf_hub_download(
        repo_id=dataset_repo_id,
        repo_type="dataset",
        filename=primary,
        local_dir=str(target_dir),
    )
    LOG.info("dataset local path: %s", local)
    return Path(local)


# ---------------------------------------------------------------------------
# Step 3: install repo into the container
# ---------------------------------------------------------------------------


def _pip_install_repo(repo_dir: Path, *, job_type: str) -> None:
    """Install repo deps. PPO needs the [ppo] extra (vllm); SFT just [train]."""

    extra = "ppo" if job_type == "ppo" else "train"
    LOG.info("installing repo deps via pip install -e .[%s]", extra)
    cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir", "-e", f".[{extra}]"]
    subprocess.check_call(cmd, cwd=str(repo_dir))


# ---------------------------------------------------------------------------
# Step 5: run the trainer
# ---------------------------------------------------------------------------


def _run_sft_trainer(*, repo_dir: Path, base_model: str, config_path: str, dataset_path: Path, output_dir: Path, output_repo: str) -> None:
    """Invoke training/sft_warmstart.py with the passed-in overrides.

    `output_repo` is the user's --output-repo from the launcher; we pass it
    to the trainer as `--hub-model-id` so each save_step pushes the LoRA
    adapter to that repo and progress survives job crashes mid-training.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "training.sft_warmstart",
        "--config",
        str(repo_dir / config_path),
        "--model",
        base_model,
        "--dataset",
        str(dataset_path),
        "--output-dir",
        str(output_dir),
        "--hub-model-id",
        output_repo,
    ]
    LOG.info("running SFT trainer: %s", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(repo_dir))


def _run_ppo_trainer(*, repo_dir: Path, base_model: str, config_path: str, output_dir: Path) -> None:
    """Run training.train_ppo with the in-job paths and base model.

    The PPO trainer reads the SFT checkpoint location from its YAML
    config (sft_checkpoint: artifacts/sft-checkpoints/final by default);
    if you need to override it, expose it via DRONECAPTUREOPS_SFT_CKPT
    and the launcher will pass --sft-checkpoint accordingly.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    mode = os.environ.get("DRONECAPTUREOPS_PPO_MODE", "train").strip().lower()
    if mode not in {"train", "preflight"}:
        raise SystemExit(f"unsupported DRONECAPTUREOPS_PPO_MODE={mode!r}; expected train or preflight")
    _wait_for_cuda_ready()

    train_ppo = repo_dir / "training" / "train_ppo.py"
    preflight_vllm = repo_dir / "training" / "ppo" / "preflight_vllm.py"
    if mode == "train" and not train_ppo.exists():
        raise SystemExit(
            "training/train_ppo.py is missing from the repo snapshot — "
            "verify the launcher packaged the latest revision."
        )
    if mode == "preflight" and not preflight_vllm.exists():
        raise SystemExit(
            "training/ppo/preflight_vllm.py is missing from the repo snapshot — "
            "verify the launcher packaged the latest revision."
        )
    if mode == "preflight":
        cmd = [
            sys.executable,
            "-m",
            "training.ppo.preflight_vllm",
            "--config",
            str(repo_dir / config_path),
            "--output",
            str(output_dir / "preflight-vllm.json"),
        ]
        LOG.info("running PPO vLLM preflight: %s", " ".join(cmd))
        subprocess.check_call(cmd, cwd=str(repo_dir))
        return

    cmd = [
        sys.executable,
        "-m",
        "training.train_ppo",
        "--config",
        str(repo_dir / config_path),
        "--model",
        base_model,
        "--output-dir",
        str(output_dir),
    ]
    sft_override = os.environ.get("DRONECAPTUREOPS_SFT_CKPT")
    if sft_override:
        cmd.extend(["--sft-checkpoint", sft_override])
    if os.environ.get("DRONECAPTUREOPS_WANDB_MODE"):
        cmd.extend(["--wandb-mode", os.environ["DRONECAPTUREOPS_WANDB_MODE"]])
    if os.environ.get("DRONECAPTUREOPS_WANDB_RUN_NAME"):
        cmd.extend(["--wandb-run-name", os.environ["DRONECAPTUREOPS_WANDB_RUN_NAME"]])

    LOG.info("running PPO trainer: %s", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(repo_dir))


# ---------------------------------------------------------------------------
# Step 6: push outputs back
# ---------------------------------------------------------------------------


def _collect_artifacts(output_dir: Path) -> list[Path]:
    """Pick out the directories worth uploading.

    By default we upload `<output_dir>/final` (trainer's saved model)
    plus anything that looks like logs / audit / metrics.
    """

    candidates: list[Path] = []
    for name in ("final", "checkpoints", "logs", "audit", "metrics", "summary"):
        path = output_dir / name
        if path.exists():
            candidates.append(path)
    if not candidates:
        # Upload everything under output_dir as a fallback.
        candidates.append(output_dir)
    return candidates


def _push_outputs(token: str, output_repo_id: str, artifact_dirs: list[Path]) -> None:
    from huggingface_hub import HfApi

    from training.hf_jobs.push_artifacts import push_training_outputs

    api = HfApi(token=token)
    push_training_outputs(api, output_repo_id=output_repo_id, artifact_dirs=artifact_dirs)


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    _configure_gpu_runtime_env()
    parser = argparse.ArgumentParser(description="In-job entrypoint for DroneCaptureOps training on HF Jobs.")
    parser.add_argument("job_type", choices=["sft", "ppo"])
    args, extra = parser.parse_known_args()

    token = _hf_token()
    repo_dataset = _required_env("DRONECAPTUREOPS_REPO_DATASET")
    output_repo = _required_env("DRONECAPTUREOPS_OUTPUT_REPO")
    base_model = _required_env("DRONECAPTUREOPS_BASE_MODEL")
    config_path = _required_env("DRONECAPTUREOPS_CONFIG")
    data_dataset = os.environ.get("DRONECAPTUREOPS_DATA_DATASET")

    git_rev = os.environ.get("DRONECAPTUREOPS_GIT_REV", "unknown")
    LOG.info("starting %s job (rev=%s, model=%s)", args.job_type, git_rev, base_model)

    # If the bootstrap shell already downloaded and extracted the repo into
    # CWD, reuse it instead of downloading again. Detect by the presence of
    # the same package layout. Falls back to the canonical download flow
    # (used by older bootstraps that didn't pre-fetch the repo).
    cwd = Path.cwd()
    if (cwd / "dronecaptureops" / "__init__.py").exists() and (cwd / "training" / "hf_jobs" / "entrypoint.py").exists():
        LOG.info("reusing pre-downloaded repo at %s (bootstrap pre-fetched)", cwd)
        repo_dir = cwd
        # Make sure WORK_ROOT exists for downstream paths (data dataset, output).
        WORK_ROOT.mkdir(parents=True, exist_ok=True)
    else:
        # Clean slate so retries are idempotent on the same container.
        if WORK_ROOT.exists():
            shutil.rmtree(WORK_ROOT, ignore_errors=True)
        WORK_ROOT.mkdir(parents=True, exist_ok=True)
        repo_dir = _download_repo(token, repo_dataset)

    _pip_install_repo(repo_dir, job_type=args.job_type)

    output_dir = WORK_ROOT / "output"
    if args.job_type == "sft":
        if data_dataset is None:
            raise SystemExit("SFT job requires DRONECAPTUREOPS_DATA_DATASET")
        data_dir = WORK_ROOT / "data"
        dataset_path = _download_dataset(token, data_dataset, data_dir)
        _run_sft_trainer(
            repo_dir=repo_dir,
            base_model=base_model,
            config_path=config_path,
            dataset_path=dataset_path,
            output_dir=output_dir,
            output_repo=output_repo,
        )
    else:
        _run_ppo_trainer(
            repo_dir=repo_dir,
            base_model=base_model,
            config_path=config_path,
            output_dir=output_dir,
        )

    artifacts = _collect_artifacts(output_dir)
    LOG.info("uploading %d artefact dir(s) to %s", len(artifacts), output_repo)
    _push_outputs(token, output_repo, artifacts)

    LOG.info("job complete; results pushed to %s", output_repo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
