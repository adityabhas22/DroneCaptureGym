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

### Cron tick 2 (~25 min after submission) — concerning
- Status: still RUNNING.
- Total log lines: 868 (unchanged from tick 1).
- vLLM emitted 15 "Processed prompts" updates and 29 "Adding requests" — engine alive.
- BUT highest completion count seen: **0/16**. ZERO rollouts have finished.
- Critic warmup step 1 hasn't completed (no `value_loss=` output).
- Throughput: ~13 toks/s output per request — way below H200's expected 50-100 toks/s for 4B.
- Hypothesis: 32k max_model_len + LoRA-on-vLLM at gpu_mem=0.4 is causing severe batch-size starvation.
- Decision: wait one more cron tick. If still no value_loss output by ~35 min in, kill + retune with smaller max_total_length and higher gpu_mem.

### Cron tick 3 (~30 min after submission) — KILLED, RCA done
- Status: cancelled.
- Root cause identified by reading log line by line:
  - vLLM engine init succeeded (LoRA hot-swap enabled, max_lora_rank=64, model loaded, KV cache allocated).
  - After init: vLLM emitted `Exception in thread Thread-5 (process_input_sockets): msgspec.ValidationError: Expected array, got int` from /usr/local/lib/python3.10/dist-packages/vllm/v1/serial_utils.py:311.
  - This is in vLLM 0.11.2's V1 engine inter-process serializer. The V1 engine became the default in vllm 0.11.0; V0 is still the LoRA-tested code path.
  - Result: rollouts silently never complete. Max completions seen across all 25 min: 0/16 (Run 1), 1/16 (Run 2). vLLM "Processed prompts" lines kept ticking but no rollout finished.
- Fix (no monkey patches):
  - pyproject [ppo] now pins `vllm>=0.10.0,<0.11.0` (forces V0 engine).
  - Belt+suspenders: set `VLLM_USE_V1=0` env in the job at submission via --env.
  - Also synced [ppo] pins with what SFT learned: peft<0.19, trl<1.2, transformers>=4.55.
- Cost burned: ~30 min × $5/hr × 2 = ~$5. Both accounts: ~$57.50 each remaining.
