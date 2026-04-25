# Run 01 (kl=0.02) decision log

## Submission
- **Job ID**: 69ed2ce5d2c8bd8662bce6ee
- **Account**: adityabhaskara
- **Submitted**: 2026-04-26 02:36:45 UTC
- **Hardware**: h200 ($5/hr)
- **Timeout**: 5h
- **Output repo**: adityabhaskara/dronecaptureops-ppo-qwen3-4b-run01-kl002
- **SFT seed**: adityabhaskara/dronecaptureops-sft-qwen3-4b:last-checkpoint (step 40 / epoch 0.74 / eval_loss 0.300)
- **Wandb run**: dronecaptureops-ppo / run01-kl002

## Decisions (chronological, populated by the babysit cron)

### Step 0 — submitted
Hypothesis: kl=0.02 keeps policy close to SFT, preserving tool-call format
while still permitting strategic improvement. Pairs with run02 (kl=0.005)
to bracket the priority dial.

### Cron tick 1 (~13 min after submission)
- Status: RUNNING. Bootstrap + pip + model load + vLLM init complete.
- Critic warmup announced: "critic warmup: 50 steps".
- Currently running rollouts for warmup (vLLM "Processed prompts" lines streaming).
- vLLM emitted one msgspec.ValidationError traceback that engine recovered from — non-fatal.
- No PPO step lines yet. Estimated first PPO step in ~15-20 min.
- No decision needed.
