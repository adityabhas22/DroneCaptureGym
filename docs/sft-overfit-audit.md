# SFT pipeline overfit audit

Goal of SFT here: teach format + rough strategy, leave enough behavioural
slack that PPO can actually explore. Overfit SFT collapses the policy onto
oracle traces, and PPO can't recover from a degenerate policy distribution.
This doc walks the pipeline and flags every knob that affects that balance.

## Pipeline shape

```
training/configs/sft_default.yaml      → generate_sft_data.py
            │                                       │
            │  (per-task seeds, policies, hold-outs) │
            ▼                                       ▼
       artifacts/sft/sft-warmstart.jsonl  ──→  training/configs/sft_train_default.yaml
                                                          │
                                                          ▼
                                                training/sft_warmstart.py (TRL SFTTrainer + LoRA)
                                                          │
                                                          ▼
                                              artifacts/sft-checkpoints/final/
```

Two configs, two scripts, one JSONL between them.

## What's already right (keep)

| Knob | Setting | Why it's correct |
|---|---|---|
| LoRA r=16, alpha=32, dropout=0.05 | trainable params ≈ 1% of base | Hard cap on memorisation surface — exactly what we want for warm-start |
| Train/val split **by seed** within task | `val_seed_fraction=0.15` | Catches within-task overfit, not just held-out-task generalisation. Critical for tool-call agents where the format is shared but the strategy varies per seed. |
| `assistant_only_loss=True` + `response_template="<|im_start|>assistant\n"` | only assistant tokens contribute to loss | Without this, the model trains on its own context (user + tool results) and learns to predict reality-as-given rather than its own actions |
| Length filter, not truncate | overlong trajectories dropped, logged | Truncating mid-tool-call corrupts the JSON in the assistant turn. Filtering is correct. |
| `early_stopping_patience=3` on `eval_loss` | stop when val plateaus | Standard anti-overfit. ToolRL/ClaimsGym both reported late epochs collapse onto memorisation. |
| 3 epochs max + cosine LR + 5% warmup | bounded fine-tuning budget | With ~few hundred examples, more epochs = pure memorisation. Cosine decay is gentler than linear at the tail. |
| `deduplicate: true` in generator | drop exact-duplicate trajectories | Oracle is deterministic given (task, seed, policy) — without dedup, repeats would inflate effective epoch count. |
| `held_out_tasks` exists | reserve some tasks for eval-only | Lets us measure cross-task generalisation cleanly. |

## What needs tweaking — concrete

### 1. `held_out_tasks` references obsolete task IDs

`training/configs/sft_default.yaml` currently lists:
```yaml
held_out_tasks:
  - multi_anomaly_triage
  - report_grounding_audit          # ← removed; new ID: audit_grade_strict_grounding
  - obstacle_detour_inspection
```

After the solar-tasks-v2 merge, only 2/3 still exist. The bad ID is silently
ignored (the resolver warns but doesn't fail). Fix the spelling.

### 2. Seed count is too low for stable val signal

`seeds_per_task: 8` × `val_seed_fraction: 0.15` = ~1 val seed per task. With
~12 train seeds × ~15 train tasks ≈ 180 train examples, val_loss has high
variance and `early_stopping_patience=3` can either trigger spuriously or
not fire at all.

**Recommendation**: bump `seeds_per_task` to **15** (val ≈ 2-3 seeds/task,
~13 train, ~195-260 train examples).

### 3. `eval_steps=25` may fire only once

With effective batch = `2 × 8 = 16` and ~200 train examples, that's
~12 steps/epoch × 3 epochs ≈ 36 total steps. `eval_steps=25` ⇒ one eval at
step 25, plus end-of-epoch evals (TRL emits those for free). Patience=3 with
that few evals never gets a chance to converge.

**Recommendation**: drop `eval_steps` to **10** so we get 3-4 evals per
epoch, ≥10 total over the run. Patience=3 then has signal.

### 4. Policy mix is too narrow

`policies: [task_oracle]` produces one trajectory per (task, seed). The
oracle is deterministic, so seed variation only changes the world (defects,
zone activations) — the *strategy* is identical across seeds for a given
task. Result: the model sees the same canonical solution repeatedly, with
only minor perturbations to the world it operates in.

**Recommendation**: add `scripted` (the weak/heuristic policy) at
`weight: 1`, conditional on it producing successful trajectories. This
gives 2 trajectories per (task, seed) where strategy actually differs.
**Caveat:** require_success: true will drop scripted trajectories that
fail, which keeps the dataset clean. We're not adding noise; we're adding
*alternative successful paths* — exactly the diversity RL needs to start
from.

If `scripted` doesn't reliably succeed on the oracle-solvable subset, omit
it; do NOT add `random` (random trajectories are noise, not diversity).

### 5. Task subset bound by oracle capability

The legacy oracle solves ~20 of 45 new tasks (subset discovered by the
agent fixing test_agent_interface.py). The other ~25 tasks have no working
reference solver. Including them would either:

- **Skip** them (`require_success: true` filters out the failures) — wastes
  generation time, no SFT signal for those task IDs.
- **Include them as failures** (if `require_success: false`) — actively
  teaches the model to fail those tasks. **Do not do this.**

**Recommendation**: explicitly list the oracle-solvable subset in
`include_tasks`, with the held-out tasks moved to `held_out_tasks`. This
makes the SFT corpus deliberate rather than accidental. The other 25 tasks
become RL-only territory — the model learns them through PPO exploration,
which is exactly the SFT→PPO division of labour we want.

### 6. Verify Qwen3-Instruct-2507 chat template

`response_template: "<|im_start|>assistant\n"` is correct for Qwen3
Instruct (and Qwen2.5) without thinking. **Do not** use it for any
Qwen3-thinking variant — those emit a `<think>...</think>` block before
the action and loss-masking on `<|im_start|>assistant\n` would include
the think block in the trained tokens, defeating the purpose. The default
`Qwen/Qwen3-4B-Instruct-2507` is the no-thinking variant, so this is fine.

Verification one-liner (run after .venv is active):
```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
print(tok.apply_chat_template(
    [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
    tokenize=False, add_generation_prompt=True,
))
```
The output must end with `<|im_start|>assistant\n` for the template to
match. If Qwen renames it in a later 2507 patch, update the YAML.

### 7. No warning on degenerate dataset shapes

`split_train_val` silently puts everything in train when a task has ≤1
seed. `filter_by_length` silently drops overlong examples. Add a guard at
the end of `split_train_val` that raises if train < 50 or val < 10 (with
val=0 only if `val_seed_fraction=0`); makes broken configs visible
immediately rather than at trainer startup.

### 8. `max_seq_length=4096` was wrong by 4× — APPLIED FIX

Measured token distribution on the generated dataset (354 oracle+scripted
trajectories, Qwen3-4B-Instruct-2507 tokenizer):

| stat | tokens |
|---|---|
| min | 9,900 |
| median | 14,531 |
| p95 | 16,366 |
| p99 | 18,886 |
| max | 19,290 |

At `max_seq_length=4096` the trainer would have silently filtered **100%**
of records. Bumped to `16384`; yield is now 96% (340 of 354 kept,
14 outliers dropped — mostly multi-anomaly variants with 30+ tool calls).

This fix also forced two related changes:
- `per_device_batch_size: 1`, `gradient_accumulation_steps: 16` (effective
  batch unchanged at 16) — 16k seq × batch 2 doesn't fit on L40S.
- `gradient_checkpointing: true` — required to keep activations under the
  L40S 48GB ceiling.

The "rich SFT data without overfit" goal is served by this — longer
trajectories with more tool calls = more strategy detail per record =
fewer epochs needed to teach the format.

## Risks I'm explicitly NOT mitigating

- **Distribution shift between SFT and PPO**. SFT teaches "what the oracle
  does on a clean simulator"; PPO will see noisy on-policy rollouts. Some
  shift is healthy — that's what the rough-strategy-not-memorisation thesis
  bets on.
- **Oracle bias**. The oracle has its own quirks (always uses the y=±24
  corridor, always submits at the end). PPO is supposed to discover better
  strategies. If the oracle is *too* good, PPO finds nothing to improve;
  if it's *too* limited, PPO has too far to climb. We accept the current
  oracle quality as fixed for this iteration.
- **No DPO/KTO step**. Pure SFT, no preference alignment. Adding DPO would
  reduce overfit further, but it's premature — we don't have preference
  labels yet, and the eval matrix is the cheaper signal.

## Applied YAML deltas (committed on this branch)

`training/configs/sft_default.yaml`:
```diff
- seeds_per_task: 8
+ seeds_per_task: 15

  policies:
    - name: task_oracle
      weight: 1
+   - name: scripted
+     weight: 1

+ include_tasks:
+   <ORACLE_SOLVABLE_TASKS list, populated after agent run>

  held_out_tasks:
    - multi_anomaly_triage
-   - report_grounding_audit
+   - audit_grade_strict_grounding
    - obstacle_detour_inspection
```

`training/configs/sft_train_default.yaml`:
```diff
- max_seq_length: 4096
+ max_seq_length: 16384

- per_device_train_batch_size: 2
- per_device_eval_batch_size: 2
- gradient_accumulation_steps: 8
+ per_device_train_batch_size: 1
+ per_device_eval_batch_size: 1
+ gradient_accumulation_steps: 16
+ gradient_checkpointing: true

- eval_steps: 25
- save_steps: 25
+ eval_steps: 10
+ save_steps: 10

- val_seed_fraction: 0.15
+ val_seed_fraction: 0.20
```

These tighten the val signal, fix the (silent) length-filter wipeout,
and add one diversity dimension via `scripted` policy in data generation,
without changing the philosophy of the pipeline (LoRA + early-stopping +
3-epoch cap + assistant-only loss all stay).

## Final dataset shape (verified)

After all deltas, with `seeds_per_task: 15` and dual oracle+scripted
policies:

| metric | value |
|---|---|
| Tasks in train set | 16 |
| Tasks held out (separate JSONL) | 4 |
| Raw trajectories generated | 600 (300 oracle + 300 scripted) |
| Kept after `require_success` | 445 (155 scripted runs failed silently — fine) |
| Kept after dedup | 354 |
| Held-out records | 91 |
| After train/val split (val_seed_fraction=0.20) | 273 train / 81 val |
| After length filter at 16384 | **262 train / 78 val** |
| Per-task train count range | 6 (diode_fault) → 20 (most) |
| Mean train trajectories/task | 16.4 |
| All trajectories successful | 354/354 (mean reward 1.000) |

This is well within the "small enough that LoRA can't memorise but big
enough that early stopping has signal" zone. The 16-task spread keeps
the per-task gradient signal balanced.
