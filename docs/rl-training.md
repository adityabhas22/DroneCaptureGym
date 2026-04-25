# Online RL training (`training/train_grpo.py`)

This document describes the GRPO/RLVR-style trainer that finishes the
RL bridge gap called out in `TEMP_NEXT_STEPS.md`. It assumes you have
already:

1. Generated SFT warm-start data with `training/generate_sft_data.py`.
2. Fine-tuned a LoRA warm-start with `training/sft_warmstart.py`.
3. Verified the SFT checkpoint with `training/eval_models.py`.

If you skip step 2 the trainer will still run, but a cold base model
rarely emits a parseable tool call, so most rollouts get parse-error
penalties and gradients get noisy fast. Warm-start strongly recommended.

---

## How it works

`training/train_grpo.py` is a small custom GRPO loop, not a wrapper
around `trl.GRPOTrainer`. We use a custom loop because our episodes are
multi-turn tool-calling sessions where one trajectory-level reward
applies to many (prompt, response) pairs — a shape `trl.GRPOTrainer`
does not natively support.

Per iteration the trainer:

1. **Picks cells.** `build_plan` samples `tasks_per_iteration` task IDs
   from `train_tasks` and `seeds_per_task_iter` seeds per task. Each
   `(task_id, seed)` is a "cell" rolled out `group_size` times.
2. **Generates rollouts.** For every cell, `LocalHFRLPolicy` drives the
   shared `RolloutRunner` through one episode using the current model
   weights. The policy records `(prompt_token_ids, response_token_ids)`
   per turn so the optimizer can recompute logits afterwards.
3. **Computes advantages.** `compute_group_advantages` takes the
   trajectory-level rewards (default: `RewardBreakdown.total`, so
   safety + integrity gates are baked in) within one cell and returns
   `(reward - mean) / std` per rollout. Same advantage applied to every
   turn of that rollout.
4. **Updates the policy.** Each turn forward-passes the
   `prompt + response` sequence, slices out the response logits, sums
   token-level log-probs, and minimises
   `-(advantage * sum_log_prob) + kl_coef * (log_p - log_p_ref)^2 / 2`.
   Gradients accumulate across `gradient_accumulation_steps` before the
   optimizer steps.
5. **Saves checkpoints.** Every `save_every_iterations` and at the end.

Reference policy (frozen base model) is loaded only when `kl_coef > 0`.

---

## Quickstart

### 1. Validate the plan without running the model

```bash
python -m training.train_grpo --dry-run --num-iterations 5 --group-size 4
```

This prints the (task_id, seed) cells the trainer would visit. No
gradients, no GPU, no model load.

### 2. Train on a tiny model end-to-end (smoke test)

```bash
python -m training.train_grpo \
  --config training/configs/rl_default.yaml \
  --num-iterations 3 --group-size 2 --no-kl
```

The default config points at `Qwen/Qwen2.5-0.5B-Instruct`, which is
small enough to run on a laptop and big enough to verify the pipeline.
Don't expect meaningful learning — the goal is "loss decreases, no
crashes, checkpoint saves succeed."

### 3. Train on a real model

```bash
python -m training.train_grpo \
  --config training/configs/rl_default.yaml \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --sft-adapter artifacts/sft-warmstart/final \
  --num-iterations 50
```

Use `--sft-adapter` to attach a LoRA adapter from `sft_warmstart.py`.
The trainer will compose your RL LoRA on top.

---

## Config

See `training/configs/rl_default.yaml` for inline comments. The high-level
knobs:

| Field | Why it matters |
| --- | --- |
| `train_tasks` | Start narrow (3-5 tasks). Expand once gates stay >0.7 average. |
| `eval_tasks` | Held-out tasks reserved for `training/eval_models.py`. |
| `group_size` | Rollouts per cell. >=4 is needed for stable group baselines. |
| `tasks_per_iteration` | Cells per iteration. Larger = more diverse data per step. |
| `kl_coef` | Penalty against the frozen base model. Higher = less drift, slower learning. |
| `reward_field` | Which `RewardBreakdown` field to optimise. Default `total` includes safety + integrity gates. |
| `advantage_normalize` | Standard GRPO; turn off for vanilla REINFORCE-style updates. |

CLI overrides take priority over the YAML config.

---

## Diagnosing a bad run

Per iteration the trainer logs JSON like:

```json
{
  "n": 8,
  "mean_reward": 0.31,
  "max_reward": 0.78,
  "min_reward": -0.4,
  "success_rate": 0.25,
  "mean_steps": 14.6,
  "parse_error_rate": 0.18,
  "mean_safety_gate": 0.92,
  "mean_integrity_gate": 0.55
}
```

Symptoms to watch for:

- **`parse_error_rate > 0.3`** — model is forgetting the tool-call
  format. Either lower `learning_rate`, raise `kl_coef`, or back off to
  more SFT.
- **`mean_integrity_gate` collapsing toward 0** — model has discovered a
  citation loophole. The new anti-gaming verifiers should already cap
  reward, but a sustained collapse means the existing tests aren't
  covering whatever pattern emerged. Investigate `citation_diagnostics`
  in the resulting trace.
- **`mean_reward` flat near 0** — group baselines may be too tight. Try
  larger `group_size` or set `advantage_normalize: false` temporarily.

The full per-iteration history is written to
`<output_dir>/training_history.json`.

---

## Limitations / known gaps

- `train()` keeps `prompt + response` in memory for every turn before
  the gradient pass. With long episodes and a 4B model that's the
  dominant memory cost; lower `group_size` or `max_steps_per_episode` if
  OOMing.
- The trainer assumes a single-GPU workflow. Multi-GPU (FSDP / DeepSpeed)
  is not wired in yet — see `training/hf_jobs/launch.py` for the hosted
  variant.
- KL is approximated via a squared sequence-level term, not exact
  per-token KL. That's enough to keep the model close to the reference
  for short rollouts; if you need a stricter KL bound, switch to TRL's
  PPOTrainer once the multi-turn reward signal is wrapped.

The next big follow-up after this trainer is the same as the rest of
the project: keep the gates honest. Add new anti-gaming tests under
`tests/test_rewards.py` whenever the trainer finds a new exploit.
