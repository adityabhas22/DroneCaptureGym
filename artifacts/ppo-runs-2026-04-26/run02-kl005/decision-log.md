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
