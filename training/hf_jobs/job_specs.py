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
# When promoting to a 14B/32B run, override on the launch CLI:
#     --hardware h200          # 1× H200, 141 GB, $5/hr
#     --hardware h200x2        # 2× H200, 282 GB, $10/hr
#     --hardware a100-large    # 1× A100, 80 GB, $2.50/hr
DEFAULT_HARDWARE_BY_JOB: dict[JobType, str] = {
    "sft": "l40sx1",    # 4B SFT (LoRA): ~25-35 min, ~$0.90 wall cost
    "ppo": "l40sx1",    # 4B PPO: ~3-4 h, ~$5-7 per lap
}
DEFAULT_TIMEOUT_BY_JOB: dict[JobType, str] = {
    "sft": "2h",        # 4B SFT lands in <40 min; 2h headroom
    "ppo": "6h",        # 4B PPO lands in ~3-4h; 6h headroom
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
    }
    if extra_env:
        env.update(extra_env)

    secrets = {
        # Both push (output upload) and pull (private dataset/model) need this.
        "HF_TOKEN": hf_token,
        "HF_AUTH_TOKEN": hf_token,
    }

    command = [
        "python",
        "-m",
        "training.hf_jobs.entrypoint",
        job_type,
    ]
    if extra_args:
        command.extend(extra_args)

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
