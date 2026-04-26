# HF Jobs wrapper

Submit DroneCaptureOps SFT and PPO training to HuggingFace Jobs (single
H200 by default; configurable). Same launcher handles both training
phases through a `--job-type` flag, so once SFT is working PPO is a
one-flag flip.

## Prerequisites

- **Pre-paid credit** on the HF account
  (https://huggingface.co/settings/billing). HF Jobs is pay-as-you-go
  for any user or organization with credits — Pro is *not* required.
- **HF token with `job.*` scopes**. Edit your token at
  https://huggingface.co/settings/tokens and grant:
  - `job.read`
  - `job.write`
  - `job.metrics.read` (optional but useful)
  Plus the usual `repo.write` so the launcher can upload code/data
  and push checkpoints to your namespace.
- **Token in env**: `HF_TOKEN` or `HF_AUTH_TOKEN`.
- **Train deps** locally for `--dry-run` smoke testing:
  `pip install -e ".[train]"`.

## Quick start

```bash
# 1. Generate SFT data locally (we still need the JSONL on disk first;
#    the launcher uploads it as a private dataset for the job).
python -m training.generate_sft_data \
    --output artifacts/sft/sft-warmstart.jsonl

# 2. Dry-run prints the resolved JobSpec without uploading or firing —
#    works without HF Pro, useful for confidence checks.
python -m training.hf_jobs.launch sft \
    --base-model Qwen/Qwen3-4B-Instruct-2507 \
    --output-repo $HF_USER/dronecaptureops-sft-qwen3-4b \
    --dry-run

# 3. Real fire (requires HF Pro). Defaults to a single H200, 4-hour
#    timeout, official transformers-pytorch-gpu image. Tail the logs
#    after submission.
python -m training.hf_jobs.launch sft \
    --base-model Qwen/Qwen3-32B \
    --output-repo $HF_USER/dronecaptureops-sft-qwen3-32b \
    --hardware h200 \
    --follow
```

PPO is the same pattern (the trainer itself is the next deliverable;
the launcher half is ready):

```bash
python -m training.hf_jobs.launch ppo \
    --base-model $HF_USER/dronecaptureops-sft-qwen3-32b \
    --output-repo $HF_USER/dronecaptureops-ppo-qwen3-32b \
    --hardware h200 \
    --follow
```

## What happens when you submit

1. **Account check** — `whoami()` and `list_jobs()`. If the latter 403s
   (account not Pro) the launcher exits with the upgrade URL.
2. **Repo snapshot** — packages the local repo (excluding `.git`,
   `.venv`, `artifacts/`, `.env*`, `__pycache__`, etc.) and uploads as
   `code.tar.gz` to a private *dataset* repo named after the base model.
3. **SFT data upload** (SFT only) — the JSONL is uploaded to a separate
   private dataset repo.
4. **Job spec built** — image, command, secrets, hardware, timeout,
   labels assembled into a `JobSpec`. Test snapshot-able without firing.
5. **Job submitted** — `huggingface_hub.run_job()`.
6. *(Inside the container)* `training/hf_jobs/entrypoint.py` downloads
   the repo, `pip install -e ".[train]"`, downloads the dataset, runs
   the trainer, then pushes the trained adapter + logs back to your
   `--output-repo`.

## Costs (rough)

The default PPO hardware is now **A100** (`a100-large`) so the first real
4B one-step smoke avoids H200 costs. For the cheapest infrastructure-only
preflight, override to L4 if available, otherwise L40S. Promote to H200
only after vLLM preflight, one-step PPO, and a tiny learning run are proven.

| Job | Model | Hardware | Wall time | Cost |
|---|---|---|---|---|
| **SFT (default)** | Qwen3-4B-Instruct-2507 (LoRA) | l40sx1 | ~30 min | ~$0.90 |
| **PPO smoke (cheap)** | Qwen3-1.7B | l4x1 or l40sx1 | <1 hr | cheapest available |
| **PPO (default)** | Qwen3-4B-Instruct-2507 | a100-large | smoke/tiny only | gated |
| SFT | Qwen3-32B (QLoRA) | h200 | ~2 hr | ~$10 |
| PPO | Qwen3-32B (QLoRA) | h200 | ~6-12 hr | ~$30-60 |

Do not run a full SFT->PPO lap until the smoke gates in
`docs/ppo-smoke-runbook.md` pass.

## Useful CLI flags

- `--dry-run`: print the resolved `JobSpec` and exit. No network calls
  to HF Hub, no uploads, no submission. Works without HF Pro.
- `--skip-push-repo` / `--skip-push-data`: re-run an iteration without
  re-uploading unchanged code/data.
- `--repo-dataset-id` / `--data-dataset-id`: pin specific Hub dataset
  IDs instead of the default per-model derivation.
- `--hardware`: pick any HF Jobs flavor (`h200`, `h200x2`, `a100-large`,
  `l40sx1`, etc.).
- `--follow`: tail job logs after submission (Ctrl-C just detaches; the
  job keeps running).
- `--extra-arg`: append a positional arg to the in-job entrypoint
  command (repeatable).

## Iterating fast

The hard-to-iterate parts of training (TRL API drift, chat template
compatibility) hit the *first* time you run a real job, regardless of
hardware. The recommended path:

1. Generate a small SFT dataset (`--seeds-per-task 2`).
2. Dry-run the launcher to confirm the spec looks right.
3. Run the Qwen3-1.7B infrastructure smoke on `--hardware l4x1` if
   available, otherwise `--hardware l40sx1`.
4. Run the first real Qwen3-4B one-step PPO smoke on `--hardware a100-large`.
5. Only after those pass, consider a tiny learning run. Keep H200 for later
   scale-up, not smoke debugging.

Step 3 is the cheap insurance for step 4.

## What's *not* yet wired

- `training/train_ppo.py` (the PPO trainer itself) — the launcher
  command path is ready, the trainer is still a TODO.
- Periodic eval inside the SFT job. Currently the trainer just saves
  and exits; eval against the saved adapter is run separately via
  `training/eval_models.py --provider hf` once the adapter is on Hub.
- Multi-GPU / DeepSpeed config for jobs >1 GPU. The wrapper accepts
  `--hardware h200x2` etc., but the trainer would need accelerate or
  deepspeed configuration to actually use the second GPU.
