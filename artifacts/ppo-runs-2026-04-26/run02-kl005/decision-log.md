# Run 02 (kl=0.005) decision log

## Submission
- **Job ID**: 69ed2cfbd70108f37acdefe3
- **Account**: G-SaiVishwas
- **Submitted**: 2026-04-26 02:37:07 UTC
- **Hardware**: h200 ($5/hr)
- **Timeout**: 5h
- **Output repo**: G-SaiVishwas/dronecaptureops-ppo-qwen3-4b-run02-kl005
- **SFT seed**: adityabhaskara/dronecaptureops-sft-qwen3-4b:last-checkpoint (now public)
- **Wandb run**: dronecaptureops-ppo / run02-kl005

## Decisions (chronological)

### Step 0 — submitted
Hypothesis: kl=0.005 lets the policy drift further from SFT, allowing
faster discovery of strategies SFT didn't cover (the 25 RL-only mechanic
families). At risk of breaking tool-call format → if parse_error_rate
climbs >10%, kill and the kl=0.02 run wins by default.

### Cron tick 1 (~13 min after submission)
- Status: RUNNING. Bootstrap + pip + model load + vLLM init complete.
- Critic warmup announced: "critic warmup: 50 steps".
- Currently running rollouts for warmup. vLLM throughput: ~3500 input toks/s, 13 output toks/s.
- Same non-fatal msgspec warning as Run 1.
- No PPO step lines yet. Estimated first PPO step in ~15-20 min.
- No decision needed.

### Cron tick 2 (~25 min after submission) — concerning (same as Run 1)
- Status: still RUNNING.
- vLLM completed 1/16 rollouts in 25 min. Same throughput problem as Run 1.
- Decision: wait one more cron tick; same kill criterion.

### Cron tick 3 (~30 min after submission) — KILLED, same RCA as Run 1
- Status: cancelled.
- Same vLLM V1 msgspec.ValidationError stalling rollouts. Same fix being applied.

### Cron tick 4 (~30 min into v2) — cancelled, same vLLM scheduler bug
- Status: cancelled (R1 hit same `max_num_batched_tokens` assertion; R2 was
  about to). Fix landed in vllm_policy.py, resubmitting v3.

### Cron tick 5 (v3 ERROR'd) — same CUDA init bug as Run 1
- Resubmitting v4 with the version bump back to 0.11.x.

### Cron tick 6 (v4 ERROR'd) — same RCA as Run 1 (see run01 log)
- v4 ERROR'd identically. Same path: vLLM 0.11.2 ignores VLLM_USE_V1=0 (V0
  was removed in 0.11.x), spawns V1 EngineCore subprocess, that subprocess
  hits Error 802 during FA3 detection at `gpu_model_runner` import time.
- Subprocess pid here: `EngineCore_DP0 pid=934`.
- Full RCA + proposed fix paths in run01-kl002/decision-log.md.
