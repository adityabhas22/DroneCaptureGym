"""Helpers for moving code, data, and checkpoints between local and HF Hub.

The launcher uses these to push the repo + SFT dataset to private Hub
repos before submitting the job. The entrypoint inside the job uses the
same helpers to download those repos and to push the trained adapter
back at the end of the run.

We always create *private* repos. The user's HF_TOKEN must have write
access. Idempotent — safe to call repeatedly with the same repo ID.
"""

from __future__ import annotations

import io
import logging
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Iterable

LOG = logging.getLogger("dronecaptureops.hf_jobs.push")


def repo_tarball_bytes(repo_root: Path, *, exclude: Iterable[str] | None = None) -> bytes:
    """Tar the repo into an in-memory blob for upload.

    Default exclusions skip everything that would bloat the upload or
    leak secrets:
        .git, .venv, __pycache__, *.pyc, artifacts/, .env, .env.*,
        *.pid, .pytest_cache, .ipynb_checkpoints, dist/, build/
    """

    excluded = set(exclude or ())
    excluded.update(
        {
            ".git",
            ".venv",
            "venv",
            "node_modules",
            "__pycache__",
            ".pytest_cache",
            ".ipynb_checkpoints",
            "artifacts",
            ".env",
            "dist",
            "build",
            ".mypy_cache",
            ".ruff_cache",
        }
    )

    def _filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
        name = Path(tarinfo.name).parts
        if any(part in excluded for part in name):
            return None
        # Exclude ANY directory whose top-level component begins with `.venv`
        # (catches `.venv-py310-test`, `.venv-cu121`, etc) or `venv-*`. Without
        # this, an ad-hoc test venv added to the repo root will bloat the
        # tarball by ~1 GB and silently hang the launcher's upload step.
        if any(p.startswith(".venv") or p.startswith("venv-") for p in name):
            return None
        if tarinfo.name.endswith(".pyc") or tarinfo.name.endswith(".pid"):
            return None
        if tarinfo.name.endswith(".env") or "/.env." in tarinfo.name:
            return None
        return tarinfo

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.add(str(repo_root), arcname=repo_root.name, filter=_filter)
    return buffer.getvalue()


def ensure_dataset_repo(api, repo_id: str, *, private: bool = True) -> None:
    """Create a private dataset repo if it doesn't exist (idempotent)."""

    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)


def ensure_model_repo(api, repo_id: str, *, private: bool = True) -> None:
    """Create a private model repo if it doesn't exist (idempotent)."""

    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)


def push_repo_tarball(
    api,
    *,
    repo_root: Path,
    dataset_repo_id: str,
    artifact_filename: str = "code.tar.gz",
) -> str:
    """Tar the repo, create the dataset repo if needed, upload the tarball.

    Returns the resolved Hub URI in `dataset_repo_id` form so the
    launcher can pass it to the job entrypoint via env var.
    """

    LOG.info("packaging %s for upload to %s", repo_root, dataset_repo_id)
    blob = repo_tarball_bytes(repo_root)
    LOG.info("packaged %.1f MB", len(blob) / (1024 * 1024))

    ensure_dataset_repo(api, dataset_repo_id)
    api.upload_file(
        path_or_fileobj=blob,
        path_in_repo=artifact_filename,
        repo_id=dataset_repo_id,
        repo_type="dataset",
        commit_message=f"upload repo snapshot ({len(blob) // 1024} KB)",
    )
    LOG.info("uploaded %s/%s", dataset_repo_id, artifact_filename)
    return dataset_repo_id


def push_jsonl_dataset(
    api,
    *,
    jsonl_path: Path,
    dataset_repo_id: str,
    path_in_repo: str | None = None,
) -> str:
    """Upload a single JSONL (e.g. the SFT dataset) to a private dataset repo."""

    if not jsonl_path.exists():
        raise FileNotFoundError(f"SFT dataset not found at {jsonl_path}")
    ensure_dataset_repo(api, dataset_repo_id)
    upload_name = path_in_repo or jsonl_path.name
    api.upload_file(
        path_or_fileobj=str(jsonl_path),
        path_in_repo=upload_name,
        repo_id=dataset_repo_id,
        repo_type="dataset",
        commit_message=f"upload SFT data ({jsonl_path.stat().st_size} bytes)",
    )
    LOG.info("uploaded %s/%s", dataset_repo_id, upload_name)
    return dataset_repo_id


def download_repo_tarball(
    api,
    *,
    dataset_repo_id: str,
    target_dir: Path,
    artifact_filename: str = "code.tar.gz",
) -> Path:
    """Download + extract the repo tarball inside the job."""

    target_dir.mkdir(parents=True, exist_ok=True)
    tar_path = api.hf_hub_download(
        repo_id=dataset_repo_id,
        repo_type="dataset",
        filename=artifact_filename,
        local_dir=str(target_dir),
    )
    LOG.info("extracted tarball from %s to %s", dataset_repo_id, target_dir)
    extract_into = target_dir / "extracted"
    extract_into.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(str(extract_into))
    # The tarball is rooted at the repo's directory name; surface it.
    candidates = [path for path in extract_into.iterdir() if path.is_dir()]
    if not candidates:
        raise RuntimeError(f"tarball at {tar_path} extracted no directories")
    return candidates[0]


def push_training_outputs(
    api,
    *,
    output_repo_id: str,
    artifact_dirs: list[Path],
    private: bool = True,
) -> str:
    """Push trained adapter + logs + audit back to a private model repo.

    Each entry in `artifact_dirs` is uploaded as a folder under the same
    name, so the resulting Hub repo looks like:
        <output_repo_id>/
            adapter/         (LoRA weights, config.json, etc.)
            logs/            (training logs, eval JSONL, summary.json)
            audit/           (per-turn raw responses if produced)
    """

    ensure_model_repo(api, output_repo_id, private=private)
    for path in artifact_dirs:
        if not path.exists():
            LOG.warning("artifact dir %s missing; skipping", path)
            continue
        api.upload_folder(
            folder_path=str(path),
            repo_id=output_repo_id,
            repo_type="model",
            path_in_repo=path.name,
            commit_message=f"upload {path.name}",
        )
        LOG.info("uploaded folder %s -> %s/%s", path, output_repo_id, path.name)
    return output_repo_id


def git_describe(repo_root: Path) -> str | None:
    """Best-effort git revision label for traceability inside the job."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return None
