"""Build HF Jobs submission specs for SFT and PPO training.

One module decides what image, what command, what secrets, and what
hardware get used per job type. Keeping this separate from the launcher
makes the wrapper testable without firing real jobs and lets us evolve
the entrypoint contract independently.

Job pipeline (both SFT and PPO):
  1. The launcher pushes the repo (as a private HF dataset) and the SFT
     dataset (also private) to HF Hub.
  2. The launcher submits a job whose command is `python -m
     training.hf_jobs.entrypoint <job_type>` with all paths/repos/configs
     in environment variables.
  3. Inside the job, the entrypoint downloads the repo+data, installs
     deps, runs the trainer, then pushes the adapter+logs+audit back
     to HF Hub.

By default we use the official HF transformers GPU image so we don't
pay 5+ minutes on every cold start to install torch/cuda. If your job
needs a newer torch, override `--image` on the launcher.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


JobType = Literal["sft", "ppo"]


# Default container — already has cuda + torch + transformers.
DEFAULT_IMAGE = "huggingface/transformers-pytorch-gpu:latest"

# Hardware tiers we routinely use, with conservative timeouts.
#
# Default flavor is L40S (48 GB VRAM, 864 GB/s memory bandwidth, ~362
# TFLOPs BF16, $1.80/hr) — the cheapest tier that comfortably fits
# Qwen3-4B-class workloads INCLUDING PPO (policy + reference + KV cache
# for rollouts) without QLoRA contortions. This is the iterate-fast-and-
# cheaply default; one full SFT→PPO lap on Qwen3-4B costs ~$6.
#
# Smoke/scale guidance:
#     --hardware l4x1          # cheapest 1.7B infra smoke, when available
#     --hardware l40sx1        # cheap fallback smoke tier
#     --hardware a100-large    # first real 4B one-step PPO smoke
#     --hardware h200          # only after smoke + tiny PPO are proven
DEFAULT_HARDWARE_BY_JOB: dict[JobType, str] = {
    "sft": "l40sx1",    # 4B SFT (LoRA): ~25-35 min, ~$0.90 wall cost
    "ppo": "a100-large", # 4B PPO smoke/default: avoid H200 until preflight passes.
}
DEFAULT_TIMEOUT_BY_JOB: dict[JobType, str] = {
    "sft": "2h",        # 4B SFT lands in <40 min; 2h headroom
    "ppo": "4h",        # Smoke/tiny PPO default; extend explicitly for larger runs.
}


@dataclass(frozen=True)
class JobSpec:
    """Materialised input to `huggingface_hub.run_job(...)`.

    Every kwarg the SDK accepts has a clearly-named attribute. The
    launcher converts this dataclass to the SDK call site; tests can
    snapshot it without firing a job.
    """

    image: str
    command: list[str]
    env: dict[str, str]
    secrets: dict[str, str]
    flavor: str
    timeout: str
    labels: dict[str, str] = field(default_factory=dict)

    def to_run_job_kwargs(self) -> dict:
        return {
            "image": self.image,
            "command": list(self.command),
            "env": dict(self.env),
            "secrets": dict(self.secrets),
            "flavor": self.flavor,
            "timeout": self.timeout,
            "labels": dict(self.labels),
        }


def build_job_spec(
    *,
    job_type: JobType,
    repo_dataset_id: str,
    data_dataset_id: str,
    output_repo_id: str,
    base_model: str,
    config_path_in_repo: str,
    hf_token: str,
    hardware: str | None = None,
    timeout: str | None = None,
    image: str = DEFAULT_IMAGE,
    extra_env: dict[str, str] | None = None,
    extra_args: list[str] | None = None,
    extra_secrets: dict[str, str] | None = None,
) -> JobSpec:
    """Build a `JobSpec` for an SFT or PPO run.

    `repo_dataset_id` — the private HF dataset holding our repo tarball.
    `data_dataset_id` — the private HF dataset holding the SFT JSONL.
    `output_repo_id` — the model repo where the trained adapter (and
        eval audit) will be uploaded at the end of the job.
    `config_path_in_repo` — relative path inside the repo to the YAML
        config the trainer reads.
    """

    if job_type not in ("sft", "ppo"):
        raise ValueError(f"unsupported job_type: {job_type!r}")

    flavor = hardware or DEFAULT_HARDWARE_BY_JOB[job_type]
    job_timeout = timeout or DEFAULT_TIMEOUT_BY_JOB[job_type]

    env = {
        "DRONECAPTUREOPS_JOB_TYPE": job_type,
        "DRONECAPTUREOPS_REPO_DATASET": repo_dataset_id,
        "DRONECAPTUREOPS_DATA_DATASET": data_dataset_id,
        "DRONECAPTUREOPS_OUTPUT_REPO": output_repo_id,
        "DRONECAPTUREOPS_BASE_MODEL": base_model,
        "DRONECAPTUREOPS_CONFIG": config_path_in_repo,
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_USE_V1": "0",
    }
    if extra_env:
        env.update(extra_env)

    secrets = {
        # Both push (output upload) and pull (private dataset/model) need this.
        "HF_TOKEN": hf_token,
        "HF_AUTH_TOKEN": hf_token,
    }
    if extra_secrets:
        secrets.update(extra_secrets)

    # The container image is plain (no repo code baked in), so we have to
    # bootstrap: fetch the repo tarball from HF, extract it, cd in, then
    # run the entrypoint. Inline this as a single `bash -lc` command so the
    # JobSpec stays self-contained — no other infrastructure to set up on the
    # cluster side. Uses `python3` (no `python` symlink in the official
    # transformers-pytorch-gpu image).
    extra_cli = " ".join(f"'{a}'" for a in (extra_args or []))
    bootstrap = (
        "set -euo pipefail; "
        "echo '[bootstrap] installing huggingface_hub'; "
        "pip install -q huggingface_hub >/dev/null 2>&1 || pip install huggingface_hub; "
        "echo '[bootstrap] downloading code tarball from '\"$DRONECAPTUREOPS_REPO_DATASET\"; "
        "python3 -c \"import os; from huggingface_hub import hf_hub_download; "
        "p = hf_hub_download(repo_id=os.environ['DRONECAPTUREOPS_REPO_DATASET'], "
        "repo_type='dataset', filename='code.tar.gz', "
        "local_dir='/workspace/_bootstrap', token=os.environ['HF_TOKEN']); "
        "print('downloaded:', p)\"; "
        "echo '[bootstrap] extracting'; "
        "mkdir -p /workspace/repo; "
        "tar -xzf /workspace/_bootstrap/code.tar.gz -C /workspace/repo; "
        "REPO_DIR=$(find /workspace/repo -maxdepth 2 -name 'pyproject.toml' -printf '%h\\n' | head -1); "
        "if [ -z \"$REPO_DIR\" ]; then echo '[bootstrap] ERROR: no pyproject.toml in extracted repo'; ls -R /workspace/repo; exit 2; fi; "
        "cd \"$REPO_DIR\"; "
        "echo '[bootstrap] running entrypoint from '\"$REPO_DIR\"; "
        f"exec python3 -m training.hf_jobs.entrypoint {job_type} {extra_cli}"
    )
    command = ["bash", "-lc", bootstrap]

    labels = {
        "project": "dronecaptureops",
        "job_type": job_type,
        "base_model": base_model.replace("/", "_"),
    }

    return JobSpec(
        image=image,
        command=command,
        env=env,
        secrets=secrets,
        flavor=flavor,
        timeout=job_timeout,
        labels=labels,
    )
