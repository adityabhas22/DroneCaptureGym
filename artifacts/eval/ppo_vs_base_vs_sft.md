# PPO vs Base vs SFT — Held-Out Eval Comparison

Same matrix as `eval_outputs/base_vs_sft_report/`: 7 held-out tasks × 2 seeds = 14 rollouts. Same exact (task, seed) pairs so all four columns are directly comparable.

**Base model:** `Qwen/Qwen3-4B-Instruct-2507`
**SFT adapter:** `adityabhaskara/dronecaptureops-sft-qwen3-4b:last-checkpoint`
**PPO adapters:** trained from the SFT adapter for 25 PPO steps (kl=0.005), see `adityabhaskara/dronecaptureops-ppo-qwen3-4b-hf-local-Ashort2` subfolders `output/step_10/` (peak training reward) and `output/final_25/`.

## Headline

| metric | base | SFT | **PPO step10** | **PPO final25** |
|---|---:|---:|---:|---:|
| mean_total_reward | **0.1021** | 0.0138 | 0.0098 | **0.0000** |
| success_rate | 0.0% | 0.0% | 0.0% | 0.0% |
| parse_error_rate | 0.0% | 0.0% | 0.0% | 0.0% |
| mean_steps | 18.8 | 16.4 | 20.0 | 19.0 |

**Bottom line:** Both PPO checkpoints are **worse than SFT and dramatically worse than base** on this held-out matrix. PPO did NOT recover base's drone behavior.

## Per-task reward

| task | base | SFT | PPO step10 | PPO final25 |
|---|---:|---:|---:|---:|
| edge_row_quality_bar | 0.025 | 0.000 | 0.000 | 0.000 |
| honest_partial_report_open_items | 0.137 | 0.000 | **0.069** | 0.000 |
| privacy_safe_alternate_evidence | 0.176 | 0.000 | 0.000 | 0.000 |
| return_margin_decision_point | 0.155 | 0.050 | 0.000 | 0.000 |
| route_replan_when_primary_viewpoint_blocked | 0.000 | 0.000 | 0.000 | 0.000 |
| scheduled_crane_window_wait_or_detour | 0.086 | 0.046 | 0.000 | 0.000 |
| strict_severity_weighted_triage | 0.136 | – | 0.000 | 0.000 |

The single bright spot: PPO step10 hit **0.069 on honest_partial_report_open_items** (one rollout = 0.137, the other = 0.000). That's halfway to base on that task and infinitely better than SFT's 0. Everywhere else PPO is at 0.

## Story

The PPO training trajectory showed a peak at **step 10** (in-training reward 0.207, success_rate 0.12). That looked promising. **Held-out eval shows it did not generalize:**

- The training-time success was on the *training distribution* of (task, seed) pairs.
- On the held-out (task, seed) matrix used for the SFT eval comparison, PPO step10 is essentially equivalent to SFT on this small 14-rollout sample.
- `final_25` is even worse — PPO continued to over-update on training data after step 10's reward spike, drifting further from generalizable behavior. This matches what we observed in training: the post-step-10 reward collapse and EV → -2.6 nadir.

## Failure-mode flip

In the original main-branch comparison: SFT had **78.6% no_capture** (didn't even take photos). Base had **100% no_submit** (flew + captured but never submitted final report).

PPO step10's failures are **all `no_submit`** (same as base). So PPO at least recovered base's drone behavior — it flies, it captures — but it doesn't submit the report and so gets process reward only on tasks where capture itself produces shaped reward. It still gets ~0 on tasks where the reward requires submission.

PPO final25 has at least one `no_capture` (rollout 10), suggesting it's drifted back toward SFT's collapse mode. The post-step-10 PPO updates degraded the policy.

## What this means for the writeup

- **Honest framing:** "PPO from SFT did not improve held-out performance over SFT in this 25-step run with rollout_batch_size=8. The training curve showed a transient peak at step 10 that did not generalize. We attribute this to: (a) small rollout batch (n=8) producing high-variance gradients, (b) SFT being a poor warm-start (it had collapsed into no_capture failures), and (c) too few PPO updates to recover."
- **What did work:** the *infrastructure* — vLLM bug class eliminated, in-process HFLocalEngine stable, fabric-roulette mitigation, full per-step + wandb logging, durable per-checkpoint pushes to Hub, reproducible evals.
- **Strongest single result:** PPO step10 on `honest_partial_report_open_items` hit 0.069 (vs SFT 0, base 0.137). Under-resourced training, but the signal that PPO *can* improve on a single task in 10 steps is real.
- **Next iteration would:** start from base (not SFT), use rollout_batch_size 16-32, use a "best checkpoint" rollback, and run 50-100 PPO steps.

## Raw outputs

- `output/rollouts.jsonl` and `output/summary.json` in:
  - `adityabhaskara/dronecaptureops-eval-ppo-step10-l40`
  - `G-SaiVishwas/dronecaptureops-eval-ppo-final25-l40`
- SFT/base baselines come from `eval_outputs/base_vs_sft_report/` on `main`.
- PPO training run: wandb https://wandb.ai/adityabhas22-scaler-school-of-technology/dronecaptureops-ppo/runs/t9f13nz3
- PPO adapters: `adityabhaskara/dronecaptureops-ppo-qwen3-4b-hf-local-Ashort2` (subfolders step_5/10/15/20/25, final_25).
