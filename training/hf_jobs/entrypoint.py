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


def _pip_install_repo(repo_dir: Path) -> None:
    LOG.info("installing repo deps via pip install -e .[train]")
    cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir", "-e", ".[train]"]
    subprocess.check_call(cmd, cwd=str(repo_dir))


# ---------------------------------------------------------------------------
# Step 5: run the trainer
# ---------------------------------------------------------------------------


def _run_sft_trainer(*, repo_dir: Path, base_model: str, config_path: str, dataset_path: Path, output_dir: Path) -> None:
    """Invoke training/sft_warmstart.py with the passed-in overrides."""

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
    ]
    LOG.info("running SFT trainer: %s", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(repo_dir))


def _run_ppo_trainer(*, repo_dir: Path, base_model: str, config_path: str, output_dir: Path) -> None:
    """Placeholder — wires up to training/train_ppo.py once that exists."""

    output_dir.mkdir(parents=True, exist_ok=True)
    train_ppo = repo_dir / "training" / "train_ppo.py"
    if not train_ppo.exists():
        raise SystemExit(
            "training/train_ppo.py is not implemented yet. "
            "PPO entrypoint plumbing is ready; the trainer itself is the next deliverable."
        )
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

    # Clean slate so retries are idempotent on the same container.
    if WORK_ROOT.exists():
        shutil.rmtree(WORK_ROOT, ignore_errors=True)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)

    repo_dir = _download_repo(token, repo_dataset)
    _pip_install_repo(repo_dir)

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
