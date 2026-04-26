# GRPO smoke and tiny-learning runbook

## What this branch is

The PPO branch was blocked by issues that all live in the colocated
vLLM rollout path: V1 thread-safety, H200 Fabric Manager error 802,
L40S OOM from 4B + vLLM colocation, and V1 ZMQ socket corruption under
multi-thread rollouts. None of those were PPO-math problems.

The `grpo/no-vllm` branch sidesteps the entire class of issues by:

- **Using GRPO instead of PPO.** No value head, no critic warmup, no
  GAE. Advantages come from group-normalized rewards within G samples
  per prompt. That deletes the L40S OOM site entirely.
- **Dropping vLLM.** Rollouts run through HF transformers
  `model.generate()` on the same in-process model. No engine-core
  process, no ZMQ socket, no fabric-manager probe needed for vLLM.
- **Single-threaded rollouts.** vLLM's V1 thread-unsafety was the root
  cause of "EngineCoreRequestType b'\x00\x00'"; HF generate is not
  thread-safe either, so the GRPO rollout pool runs episodes one at a
  time. Total rollouts per step are tiny anyway (4 prompts x 4 group =
  16) so the throughput hit is acceptable.

We pay for this in generation speed (HF generate ~30 to 60 toks/s on
L40S vs vLLM ~100 to 200 toks/s). For a 10-step tiny learning run that
is acceptable.

## Modules added

- `dronecaptureops/agent/hf_generate_policy.py`: Policy that calls
  `model.generate()` directly. Same chat-template / system-prompt /
  tool-schema construction as `VLLMPolicy`.
- `training/grpo/config.py`: `GRPOTrainConfig` Pydantic schema with
  `prompts_per_step`, `group_size`, `kl_coef`, `clip_eps`, no
  value/critic fields, no vLLM fields.
- `training/grpo/loss.py`: `group_advantages` + `grpo_policy_loss`,
  re-uses `compute_approx_kl`, `gather_token_log_probs`, `masked_mean`
  from `training/ppo/loss.py`.
- `training/grpo/rollout_pool.py`: Sequential, single-threaded rollout
  collector. Dropped rollouts are surfaced as `None` so the trainer
  can decide to skip the group rather than crash.
- `training/grpo/trainer.py`: 1-GPU GRPO trainer. Loads base + SFT
  LoRA, single optimizer (no value head), CUDA fail-fast guard, JSONL
  metrics + per-step Markdown report.
- `training/grpo/reporting.py`: Thin wrapper around the existing PPO
  reporting helpers; adds GRPO-flavoured per-step Markdown showing
  group-level reward spread.
- `training/train_grpo.py`: CLI entrypoint mirroring `train_ppo.py`.
  Same `_validate_task_ids` gate, `--dry-run`, `--allow-fresh-lora`,
  `--no-fit`.
- `training/hf_jobs/{job_specs,entrypoint,launch}.py`: extended to
  support `job_type=grpo` with `l40sx1` default, 4h timeout, `[train]`
  pip extras (no vLLM dependency).

## Configs

- `training/configs/grpo_smoke_4b_l40.yaml`: 1 prompt, group_size=2,
  total_steps=1. Mechanics-only smoke. `allow_fresh_lora=false`.
- `training/configs/grpo_tiny_4b_l40.yaml`: 4 prompts/step,
  group_size=4, total_steps=10. KL=0.01.
- `training/configs/grpo_tiny_4b_l40_kl005.yaml`: same shape, KL=0.005.
- `training/configs/grpo_tiny_4b_l40_kl02.yaml`: same shape, KL=0.02.

## Phases

### Phase 0: Local dry-run

```bash
python -m training.train_grpo --config training/configs/grpo_smoke_4b_l40.yaml --dry-run
python -m training.train_grpo --config training/configs/grpo_tiny_4b_l40_kl005.yaml --dry-run
python -m training.train_grpo --config training/configs/grpo_tiny_4b_l40_kl02.yaml --dry-run
```

Should print resolved config + `n_train_tasks > 0`. No GPU touched.
Existing test suite covers this.

### Phase 1: Remote smoke (L40S)

```bash
python -m training.hf_jobs.launch grpo \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --output-repo <namespace>/dronecaptureops-grpo-smoke-4b-l40 \
  --config training/configs/grpo_smoke_4b_l40.yaml \
  --hardware l40sx1 \
  --timeout 2h
```

Goal: prove the no-vLLM HF-generate path runs end-to-end on L40S.

Success gates (in order):

1. CUDA readiness probe passes.
2. SFT LoRA loads strictly (no fresh-LoRA fallback).
3. One step completes: `metrics.jsonl` has at least one row.
4. `step_1` checkpoint uploaded to the output repo.
5. `reports/training_report.md` is non-empty.

Kill criteria:

- Any GPU OOM during rollout or update.
- Probe exhausts its 30-attempt budget and CUDA is still not ready.
- `n_assistant_turns == 0` for every rollout (chat template or model
  load is broken).

### Phase 2: Remote tiny learning, KL sweep (L40S)

Only after Phase 1 is green.

```bash
# Branch A: low KL (more exploration off SFT).
python -m training.hf_jobs.launch grpo \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --output-repo <namespace>/dronecaptureops-grpo-tiny-4b-l40-kl005 \
  --config training/configs/grpo_tiny_4b_l40_kl005.yaml \
  --hardware l40sx1 \
  --timeout 4h

# Branch B: higher KL (closer to SFT, safer format).
python -m training.hf_jobs.launch grpo \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --output-repo <namespace>/dronecaptureops-grpo-tiny-4b-l40-kl02 \
  --config training/configs/grpo_tiny_4b_l40_kl02.yaml \
  --hardware l40sx1 \
  --timeout 4h
```

Goal: produce the KL comparison the PPO branch never reached.

Expected wallclock: 1.5 to 2 h per branch on L40S with HF generate at
this batch shape (10 steps, 16 rollouts/step, ~8 turns each).

Success gates per branch:

1. `metrics.jsonl` has 10 rows.
2. `parse_error_rate` stays roughly constant or drops over the run.
3. `kl_to_ref` does not blow up; sane values are O(0.01) to O(0.5).
4. `mean_total_reward` shows directionality across steps (does not
   need to monotonically improve, but should not crash to zero with
   high parse errors).
5. `reports/charts/reward_success.svg` and `failure_modes.svg` are
   written.

Kill criteria:

- `parse_error_rate > 0.5` for two consecutive steps -> kill, the
  policy is breaking format. Lower KL coefficient is the likely cause.
- `kl_to_ref > 5.0` for any step -> kill, drift too aggressive.
- Any uncaught training exception -> kill and root-cause.

## Risks and mitigations

- **HF generate is slow.** Acceptable for 10-step tiny runs. If we
  want >50 steps later, reintroduce vLLM only after the GRPO core is
  proven and the V1 thread-safety / Fabric Manager mitigations from
  the PPO branch are confirmed working again.
- **Memory still tight on L40S.** If 4B + group_size=4 OOMs, fall back
  to group_size=2 in the smoke config and try again.
- **PEFT `disable_adapter()` is required for ref logprobs.** Same
  pattern the PPO trainer uses; if PEFT cannot disable the adapter,
  ref logprobs will silently come from the current policy (zero KL
  signal).
- **No HF GRPO/transformers upgrade.** We keep existing pins
  (`trl<1.2`, `transformers>=4.55`), so the SFT pipeline cannot break.
