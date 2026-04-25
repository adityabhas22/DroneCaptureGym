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
