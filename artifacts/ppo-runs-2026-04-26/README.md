# PPO Sweep — overnight run, 2026-04-26

Two parallel H200 PPO runs continuing from the SFT adapter at
`adityabhaskara/dronecaptureops-sft-qwen3-4b:last-checkpoint` (eval_loss
0.300 at step 40 / epoch 0.74).

## Sweep design

The runbook calls out `kl_coef` as the most-tuned dial. With only 2 parallel
slots, sweep the *informative extremes* (4× ratio) rather than adjacent
values, so one clear winner emerges.

| Run | Account | kl_coef | Hypothesis |
|---|---|---|---|
| run01 | adityabhaskara | 0.02 | Conservative; tool-call format protected; slower drift but stable |
| run02 | G-SaiVishwas | 0.005 | Exploratory; faster strategy discovery; risks parse-error explosion |

Other knobs held constant per the default config:
- `actor_lr=1e-5`, `critic_lr=1e-5`
- `ppo_epochs=2`, `eps_clip_low/high=0.2/0.2`, `value_clip=0.2`, `lam=0.95`, `gamma=0.99`
- `lora_rank=64, alpha=32, dropout=0.05`
- `rollout_batch_size=16`, `temperature=0.7`, `max_total_length=32768`
- `entropy_coef=0.001`
- `save_interval_steps=10` (5 checkpoints/run, recoverable on crash; CHANGED from default 25)
- Run 2 uses `seed=17` (Run 1 uses 42) so rollouts diverge between runs

## Reproduction

Each run dir contains the EXACT YAML config used. To reproduce a run:

```
git checkout $(cat git-rev.txt)
python -m training.hf_jobs.launch ppo \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --output-repo <your-namespace>/dronecaptureops-ppo-replay \
  --config artifacts/ppo-runs-2026-04-26/run01-kl002/config-snapshot.yaml \
  --hardware h200 \
  --env DRONECAPTUREOPS_SFT_CKPT=adityabhaskara/dronecaptureops-sft-qwen3-4b:last-checkpoint \
  --env DRONECAPTUREOPS_WANDB_MODE=online
```

## Decisions log

See `decision-log.md` (per run) for every kill / extend / retune decision
and the metrics that triggered it.

## Final summary

After both runs complete (or are killed), `summary.md` in this dir compares
metrics and recommends the best config for any follow-on run.
