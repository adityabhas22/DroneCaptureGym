"""Tests for the HF Jobs launcher (mocked, no network calls)."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from training.hf_jobs.job_specs import (
    DEFAULT_HARDWARE_BY_JOB,
    DEFAULT_IMAGE,
    JobSpec,
    build_job_spec,
)
from training.hf_jobs.push_artifacts import (
    download_repo_tarball,
    ensure_dataset_repo,
    git_describe,
    push_jsonl_dataset,
    push_repo_tarball,
    push_training_outputs,
    repo_tarball_bytes,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# JobSpec construction
# ---------------------------------------------------------------------------


def test_build_job_spec_sft_defaults():
    spec = build_job_spec(
        job_type="sft",
        repo_dataset_id="user/repo-bundle",
        data_dataset_id="user/sft-data",
        output_repo_id="user/sft-out",
        base_model="Qwen/Qwen3-4B-Instruct-2507",
        config_path_in_repo="training/configs/sft_train_default.yaml",
        hf_token="hf_fake",
    )
    assert spec.image == DEFAULT_IMAGE
    assert spec.flavor == DEFAULT_HARDWARE_BY_JOB["sft"]
    # Default flavor is the cheap-iteration tier (L40S) for 4B-class workloads.
    assert spec.flavor == "l40sx1"
    assert spec.timeout == "2h"
    assert spec.command == ["python", "-m", "training.hf_jobs.entrypoint", "sft"]
    # Both env names get the secret so legacy code paths work.
    assert spec.secrets == {"HF_TOKEN": "hf_fake", "HF_AUTH_TOKEN": "hf_fake"}
    # Critical env vars present.
    assert spec.env["DRONECAPTUREOPS_JOB_TYPE"] == "sft"
    assert spec.env["DRONECAPTUREOPS_REPO_DATASET"] == "user/repo-bundle"
    assert spec.env["DRONECAPTUREOPS_DATA_DATASET"] == "user/sft-data"
    assert spec.env["DRONECAPTUREOPS_OUTPUT_REPO"] == "user/sft-out"
    assert spec.env["DRONECAPTUREOPS_BASE_MODEL"] == "Qwen/Qwen3-4B-Instruct-2507"
    # Labels for filtering jobs in the HF UI later.
    assert spec.labels["project"] == "dronecaptureops"
    assert spec.labels["job_type"] == "sft"


def test_build_job_spec_ppo_uses_longer_timeout():
    spec = build_job_spec(
        job_type="ppo",
        repo_dataset_id="user/repo",
        data_dataset_id="user/data",
        output_repo_id="user/out",
        base_model="Qwen/Qwen3-32B",
        config_path_in_repo="training/configs/ppo_train_default.yaml",
        hf_token="hf_fake",
    )
    # PPO timeout is longer than SFT (rollouts + value-head training).
    assert spec.timeout == "6h"
    assert spec.command[-1] == "ppo"


def test_build_job_spec_hardware_override():
    spec = build_job_spec(
        job_type="sft",
        repo_dataset_id="user/repo",
        data_dataset_id="user/data",
        output_repo_id="user/out",
        base_model="Qwen/Qwen3-32B",
        config_path_in_repo="x.yaml",
        hf_token="hf_fake",
        hardware="h200x2",
        timeout="8h",
    )
    assert spec.flavor == "h200x2"
    assert spec.timeout == "8h"


def test_build_job_spec_rejects_unknown_job_type():
    with pytest.raises(ValueError, match="unsupported job_type"):
        build_job_spec(
            job_type="bogus",  # type: ignore[arg-type]
            repo_dataset_id="x",
            data_dataset_id="y",
            output_repo_id="z",
            base_model="m",
            config_path_in_repo="c.yaml",
            hf_token="t",
        )


def test_job_spec_to_run_job_kwargs_round_trips():
    spec = JobSpec(
        image="img",
        command=["python", "-m", "x"],
        env={"A": "1"},
        secrets={"HF_TOKEN": "t"},
        flavor="h200",
        timeout="4h",
        labels={"k": "v"},
    )
    kwargs = spec.to_run_job_kwargs()
    assert kwargs["image"] == "img"
    assert kwargs["command"] == ["python", "-m", "x"]
    assert kwargs["env"] == {"A": "1"}
    assert kwargs["secrets"] == {"HF_TOKEN": "t"}
    assert kwargs["flavor"] == "h200"
    assert kwargs["timeout"] == "4h"
    assert kwargs["labels"] == {"k": "v"}


# ---------------------------------------------------------------------------
# Tarball helpers
# ---------------------------------------------------------------------------


def test_repo_tarball_bytes_excludes_secrets_and_caches(tmp_path: Path):
    """Tar should drop .git, .venv, artifacts, .env, __pycache__, .pyc files."""

    (tmp_path / "code.py").write_text("print('hi')\n")
    (tmp_path / "README.md").write_text("# hi\n")

    # Stuff that MUST NOT end up in the tarball.
    (tmp_path / ".env").write_text("HF_AUTH_TOKEN=hf_secret_xxx\n")
    (tmp_path / ".env.local").write_text("OPENAI_KEY=sk_secret_xxx\n")
    (tmp_path / "code.pyc").write_text("byte-compiled junk")
    (tmp_path / "run.pid").write_text("12345")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "binary").write_text("torch wheel goes here")
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / "logs.txt").write_text("private log")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "x.pyc").write_text("cached")

    blob = repo_tarball_bytes(tmp_path)
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        names = tar.getnames()

    joined = "\n".join(names)
    # Must include legitimate code.
    assert any(name.endswith("code.py") for name in names)
    assert any(name.endswith("README.md") for name in names)
    # Must NOT include any of these.
    assert ".env" not in joined
    assert ".env.local" not in joined
    assert ".git" not in joined
    assert ".venv" not in joined
    assert "artifacts" not in joined
    assert "__pycache__" not in joined
    assert ".pyc" not in joined
    assert ".pid" not in joined


def test_push_repo_tarball_uploads_blob_and_creates_repo():
    api = MagicMock()
    repo_root = REPO_ROOT  # use the actual repo so we know the tarball is non-empty

    result = push_repo_tarball(api, repo_root=repo_root, dataset_repo_id="user/bundle")

    assert result == "user/bundle"
    api.create_repo.assert_called_once()
    create_kwargs = api.create_repo.call_args.kwargs
    assert create_kwargs["repo_id"] == "user/bundle"
    assert create_kwargs["repo_type"] == "dataset"
    assert create_kwargs["private"] is True
    assert create_kwargs["exist_ok"] is True

    api.upload_file.assert_called_once()
    upload_kwargs = api.upload_file.call_args.kwargs
    assert upload_kwargs["path_in_repo"] == "code.tar.gz"
    assert upload_kwargs["repo_type"] == "dataset"
    assert upload_kwargs["repo_id"] == "user/bundle"
    # path_or_fileobj is the tarball blob (bytes).
    assert isinstance(upload_kwargs["path_or_fileobj"], (bytes, bytearray))
    assert len(upload_kwargs["path_or_fileobj"]) > 1024


def test_push_jsonl_dataset_requires_existing_file(tmp_path: Path):
    api = MagicMock()
    missing = tmp_path / "nope.jsonl"
    with pytest.raises(FileNotFoundError):
        push_jsonl_dataset(api, jsonl_path=missing, dataset_repo_id="user/data")


def test_push_jsonl_dataset_uploads_when_present(tmp_path: Path):
    api = MagicMock()
    path = tmp_path / "sft.jsonl"
    path.write_text(json.dumps({"task_id": "x", "messages": []}) + "\n")
    push_jsonl_dataset(api, jsonl_path=path, dataset_repo_id="user/data")
    api.create_repo.assert_called_once()
    api.upload_file.assert_called_once()
    assert api.upload_file.call_args.kwargs["path_in_repo"] == "sft.jsonl"


def test_push_training_outputs_skips_missing_dirs(tmp_path: Path):
    api = MagicMock()
    real = tmp_path / "final"
    real.mkdir()
    (real / "adapter.bin").write_text("weights")
    fake = tmp_path / "logs"  # does not exist
    push_training_outputs(api, output_repo_id="user/out", artifact_dirs=[real, fake])
    # create_repo always called; upload_folder only for the existing dir.
    api.create_repo.assert_called_once()
    assert api.upload_folder.call_count == 1
    assert api.upload_folder.call_args.kwargs["folder_path"] == str(real)


# ---------------------------------------------------------------------------
# Account verification
# ---------------------------------------------------------------------------


def test_verify_hf_access_clear_error_when_token_lacks_job_scopes():
    """403 with 'missing permissions: job.read' must point at the token-scope fix."""

    from training.hf_jobs.launch import AccountError, verify_hf_access

    fake_api = MagicMock()
    fake_api.whoami.return_value = {"name": "u", "type": "user", "isPro": False}
    fake_api.list_jobs.side_effect = Exception(
        "403 Forbidden: You don't have the required permissions, missing permissions: job.read"
    )
    with patch("huggingface_hub.HfApi", return_value=fake_api):
        with pytest.raises(AccountError, match=r"job\.\* fine-grained scopes"):
            verify_hf_access("hf_fake")


def test_verify_hf_access_generic_error_for_unknown_failure():
    """Other list_jobs failures (e.g. no credits) get the generic actionable message."""

    from training.hf_jobs.launch import AccountError, verify_hf_access

    fake_api = MagicMock()
    fake_api.whoami.return_value = {"name": "u", "type": "user", "isPro": False}
    fake_api.list_jobs.side_effect = Exception("500 Internal Server Error")
    with patch("huggingface_hub.HfApi", return_value=fake_api):
        with pytest.raises(AccountError, match="Could not list HF Jobs"):
            verify_hf_access("hf_fake")


def test_verify_hf_access_succeeds_when_jobs_listing_works():
    from training.hf_jobs.launch import verify_hf_access

    fake_api = MagicMock()
    fake_api.whoami.return_value = {"name": "u", "type": "user", "isPro": True}
    fake_api.list_jobs.return_value = []
    with patch("huggingface_hub.HfApi", return_value=fake_api):
        whoami = verify_hf_access("hf_fake")
    assert whoami["name"] == "u"


# ---------------------------------------------------------------------------
# Submission path with mocked run_job
# ---------------------------------------------------------------------------


def test_submit_calls_run_job_with_job_spec_kwargs():
    from training.hf_jobs.launch import submit

    spec = build_job_spec(
        job_type="sft",
        repo_dataset_id="user/repo",
        data_dataset_id="user/data",
        output_repo_id="user/out",
        base_model="Qwen/Qwen3-4B-Instruct-2507",
        config_path_in_repo="x.yaml",
        hf_token="hf_fake",
    )
    fake_info = SimpleNamespace(id="job-123")
    with patch("huggingface_hub.run_job", return_value=fake_info) as mocked:
        job_id = submit(spec, token="hf_fake")
    assert job_id == "job-123"
    mocked.assert_called_once()
    kwargs = mocked.call_args.kwargs
    assert kwargs["image"] == DEFAULT_IMAGE
    # SFT default flavor is L40S (the cheap-iteration tier).
    assert kwargs["flavor"] == "l40sx1"
    assert kwargs["secrets"]["HF_TOKEN"] == "hf_fake"
    assert kwargs["token"] == "hf_fake"


def test_submit_dry_run_does_not_call_run_job(capsys):
    from training.hf_jobs.launch import submit

    spec = build_job_spec(
        job_type="sft",
        repo_dataset_id="user/repo",
        data_dataset_id="user/data",
        output_repo_id="user/out",
        base_model="x",
        config_path_in_repo="c.yaml",
        hf_token="hf_fake",
    )
    with patch("huggingface_hub.run_job") as mocked:
        result = submit(spec, token="hf_fake", dry_run=True)
    assert result is None
    mocked.assert_not_called()


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_launch_dry_run_cli_succeeds(tmp_path: Path):
    """End-to-end CLI dry-run: no network calls, exits 0, prints a spec."""

    sft = tmp_path / "sft.jsonl"
    sft.write_text(json.dumps({"task_id": "basic_thermal_survey", "messages": []}) + "\n")

    proc = subprocess.run(
        [
            sys.executable, "-m", "training.hf_jobs.launch", "sft",
            "--base-model", "Qwen/Qwen3-4B-Instruct-2507",
            "--output-repo", "u/sft-out",
            "--sft-data", str(sft),
            "--dry-run",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env={**__import__("os").environ, "HF_AUTH_TOKEN": "hf_fake_for_dry_run"},
    )
    assert proc.returncode == 0, proc.stderr
    combined = proc.stdout + proc.stderr
    assert "DRY RUN" in combined
    # Default flavor for SFT is L40S (cheap-iteration tier).
    assert "l40sx1" in combined
    assert "Qwen/Qwen3-4B-Instruct-2507" in combined
    assert "<redacted>" in combined  # secret not leaked


# ---------------------------------------------------------------------------
# git_describe
# ---------------------------------------------------------------------------


def test_git_describe_returns_short_hash_for_real_repo():
    rev = git_describe(REPO_ROOT)
    # Either a short SHA or None; in CI without git it could be None,
    # but in this checkout it'll be a hash.
    assert rev is None or len(rev) >= 4
