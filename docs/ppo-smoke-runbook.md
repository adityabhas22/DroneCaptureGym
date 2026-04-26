# PPO smoke runbook

Use this runbook before spending on any full PPO sweep. The current rule is:

1. Prove cheap rollout infrastructure with Qwen3-1.7B on L4 if available, otherwise L40S.
2. Prove one real PPO step with Qwen3-4B on A100.
3. Only then run a tiny learning job.
4. Do not use H200 or launch a full KL sweep until those gates pass.

## Token handling

Local CLIs load simple `KEY=VALUE` entries from `.env` without printing secret values. The relevant token names are:

- `HF_TOKEN`
- `HF_AUTH_TOKEN`
- `HUGGINGFACE_TOKEN`
- `HUGGING_FACE_HUB_TOKEN`

Logs only report whether a token is visible and which variable name was used.

## Local sanity

From the repo root:

```powershell
python -m pytest
python examples/run_scripted_agent.py
python examples/run_task_suite.py
python -m training.run_suite --suite smoke --policy random
python -m training.run_suite --suite smoke --policy weak_scripted
python -m training.run_suite --suite smoke --policy scripted
python -m training.train_ppo --dry-run
```

Expected baseline shape:

- random: near-zero reward, zero success
- weak scripted: partial reward, zero success
- scripted: reward near 1.0, success near 1.0

## Cheap infrastructure smoke

Config:

`training/configs/ppo_smoke_1p7b_l4.yaml`

This uses `Qwen/Qwen3-1.7B` and explicitly allows fresh LoRA. It is infrastructure-only because there is no matching SFT adapter configured.

Dry-run:

```powershell
python -m training.train_ppo --config training/configs/ppo_smoke_1p7b_l4.yaml --dry-run
```

Local/GPU preflight command:

```powershell
python -m training.ppo.preflight_vllm --config training/configs/ppo_smoke_1p7b_l4.yaml --output artifacts/ppo-smoke-1p7b/preflight-vllm.json
```

HF Jobs dry-run:

```powershell
python -m training.hf_jobs.launch ppo `
  --base-model Qwen/Qwen3-1.7B `
  --output-repo <namespace>/dronecaptureops-ppo-smoke-1p7b `
  --config training/configs/ppo_smoke_1p7b_l4.yaml `
  --hardware l4x1 `
  --timeout 1h `
  --dry-run
```

If `l4x1` is unavailable, use `--hardware l40sx1`.

## First real one-step PPO smoke

Config:

`training/configs/ppo_smoke_4b_a100.yaml`

This uses `Qwen/Qwen3-4B-Instruct-2507` and requires:

`adityabhaskara/dronecaptureops-sft-qwen3-4b:last-checkpoint`

Dry-run:

```powershell
python -m training.train_ppo --config training/configs/ppo_smoke_4b_a100.yaml --dry-run
```

Preflight:

```powershell
python -m training.ppo.preflight_vllm --config training/configs/ppo_smoke_4b_a100.yaml --output artifacts/ppo-smoke-4b/preflight-vllm.json
```

HF Jobs dry-run:

```powershell
python -m training.hf_jobs.launch ppo `
  --base-model Qwen/Qwen3-4B-Instruct-2507 `
  --output-repo <namespace>/dronecaptureops-ppo-smoke-4b `
  --config training/configs/ppo_smoke_4b_a100.yaml `
  --hardware a100-large `
  --timeout 2h `
  --dry-run
```

Only remove `--dry-run` after the job spec looks correct.

## vLLM triage order

Run preflight only. Do not run PPO while triaging vLLM.

1. Default path: do not set `VLLM_WORKER_MULTIPROC_METHOD=spawn`.
2. If CUDA/FlashAttention init fails, try `VLLM_FLASH_ATTN_VERSION=2`.
3. If rollout still stalls, keep context at 8192, rollout batch size 1, and max workers 1.
4. If vLLM still fails, design a raw-transformers fallback separately.

## One-step PPO success criteria

A one-step smoke passes only if:

- rollout batch completes,
- tokenization succeeds,
- reward placement succeeds,
- one PPO update runs,
- `metrics.jsonl` has a record,
- checkpoint saves,
- losses/KL/entropy are finite,
- parse error rate is logged,
- no CUDA Error 802,
- no repeated `msgspec.ValidationError`,
- no stalled vLLM processed-prompt logs with 0 completed rollouts.

## Kill criteria

Stop immediately if:

- no rollout completes in the expected smoke window,
- vLLM logs activity but completion count stays at 0,
- CUDA Error 802 appears,
- `msgspec.ValidationError` repeats,
- no metrics file appears after the first expected PPO step,
- policy loss, value loss, or KL is NaN/inf,
- parse error rate exceeds 20% on easy smoke tasks,
- checkpoint saving fails.

## Scaling

After the 4B/A100 one-step smoke passes, create a tiny 4B/A100 learning config:

- 5 to 10 PPO steps
- rollout batch size 2 to 4
- max workers 1 to 2
- easy tasks only
- save every 5 steps
- eval every 5 steps if vLLM eval is stable

Only after that tiny run completes should a reduced KL comparison run:

- `kl_coef=0.02`
- `kl_coef=0.005`

Do not start with 50 steps or H200.
