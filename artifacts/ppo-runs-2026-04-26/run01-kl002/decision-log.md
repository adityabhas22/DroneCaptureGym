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

### Cron tick 4 (~30 min into v2) — KILLED, RCA #2
- Status: ERROR (R1), cancelled R2 (was crashing same way).
- vLLM V0 engine confirmed working (no msgspec errors). But:
  - Innermost error: `vllm/core/scheduler.py:1268` →
    `assert budget.num_batched_tokens <= max_num_batched_tokens`
  - We never set `max_num_batched_tokens` on the LLM(); vLLM defaults small
    (~2048) which is way under our 32k prompts. First long-prompt rollout
    triggers the assertion → IndexError downstream → process dies.
- Fix in dronecaptureops/agent/vllm_policy.py: explicitly pass
  `max_num_batched_tokens=max_model_len` so any single prompt fits.
- Cost burned this attempt: ~30 min × $5/hr × 2 = ~$5 (cumulative across
  v1+v2 attempts: ~$10).
- Resubmitting v3 with this fix.

### Cron tick 5 (v3 ERROR'd) — RCA #3
- Both v3 runs ERROR'd during vLLM init.
- Innermost error:
  `RuntimeError: Unexpected error from cudaGetDeviceCount(). Error 802: system not yet initialized`
- Stack: vllm.executor.uniproc_executor → init_device → torch.cuda.set_device → cuda_init → Error 802.
- Root cause: vLLM 0.10.x's uniproc_executor init flow is incompatible with our setup
  (PyTorch loads + initializes CUDA for the trainable model BEFORE vLLM tries to start
  its inference worker). vLLM 0.11.2 (which we used in v2) handled this — its V0 engine
  init worked cleanly.
- Fix: bump pin BACK to `vllm>=0.11.0,<0.12.0`, keep VLLM_USE_V1=0 to avoid msgspec.
- The max_num_batched_tokens=max_model_len fix from v3 STAYS — that addresses the V0
  scheduler assertion v2 hit during rollouts.
- Cumulative cost burned: ~$15. Budget remaining: ~$75 across 3 accounts.
- Resubmitting v4 with vLLM 0.11.2 + V0 forced + max_num_batched_tokens fix.

### Cron tick 6 (v4 ERROR'd) — RCA #4 — ESCALATING (4 attempts at init)
- Both v4 runs ERROR'd identically. Initial reading was "shifted earlier" CUDA
  init failure, but full log trace told a different story.
- Real root cause (read from `/tmp/run01_v4.log` lines 631–769):
  - vLLM 0.11.2 announces: `Initializing a V1 LLM engine (v0.11.2)` despite
    `VLLM_USE_V1=0` being set in the env. Reason: **vllm 0.11.x removed the V0
    engine entirely** — the env var still exists but no longer has a backend
    to switch to. This is a real regression from v2's behavior (which ran
    vllm 0.10.x where V0 was real).
  - V1 engine spawns `EngineCore_DP0` worker subprocess (we set
    `VLLM_WORKER_MULTIPROC_METHOD=spawn` per the v4 launcher header).
  - The spawned subprocess imports `vllm.v1.worker.gpu_model_runner` →
    transitively imports `vllm.v1.attention.backends.flash_attn`. At
    *class-body* evaluation time of `FlashAttentionMetadataBuilder` (line 230),
    vLLM calls `get_flash_attn_version()` to decide between FA2 and FA3.
  - That call path: `fa_utils.py:41` → `flash_attn_interface.py:51` →
    `torch.cuda.get_device_capability(device)` → `torch.cuda._lazy_init()` →
    `RuntimeError: cudaGetDeviceCount() Error 802: system not yet initialized`.
  - Why Error 802 in spawn mode: the parent process initialized CUDA when
    loading the trainable Qwen model. The spawned subprocess can't safely
    re-init CUDA in this HF Jobs container — `spawn` doesn't carry CUDA
    state, and import-time `_lazy_init` fires before the worker has a
    chance to set up its own context.
- The earlier line-568 warning about CUDA init from the parent process was a
  red herring — the parent recovered, loaded the model on GPU successfully,
  and started vLLM. The fatal error is in the spawned worker, not the parent.
- This explains everything across v1–v4:
  - v1 (vllm 0.11.2 default V1): same V1 path but msgspec error masked the FA3 issue.
  - v2 (vllm 0.10.x, V0): real V0 → no spawn subprocess → CUDA init clean → only
    hit the scheduler assertion.
  - v3 (vllm 0.10.x, V0): same V0 path but earlier crash in uniproc_executor's
    own `set_device` (a 0.10.x quirk we can't patch around without forking vLLM).
  - v4 (vllm 0.11.2, V1 forced + spawn): hit the FA3 import-time CUDA init.
- Fix paths (NOT submitted yet — escalating):
  1. **Drop `VLLM_WORKER_MULTIPROC_METHOD=spawn`**: let vllm V1 default to fork
     (single-GPU). Fork inherits the parent's CUDA context, so the FA3
     detection can read device capability from the inherited state. Cheapest
     possible fix — 1 env var change, no code change.
  2. **Set `VLLM_FLASH_ATTN_VERSION=2`**: tells vllm to skip the FA3 detection
     path entirely. Avoids the import-time CUDA call. Slight perf cost (FA2 vs
     FA3 on Hopper) but irrelevant for our 4B model.
  3. **Cut vLLM**: use raw `transformers.generate()` for rollouts. User
     suggested this earlier. Throughput drops 5–10× but eliminates ALL vLLM
     init pathology. Probably can't fit 50 PPO steps × 16 rollouts in
     remaining budget.
  4. **Pin vllm to 0.11.0 or 0.11.1**: 0.11.2 may have introduced the
     class-body FA3 detection. Older 0.11.x might do it lazily.
- Recommendation: try (1) first ($5 to test), then (2) if it fails ($5 more),
  then (3) only if both fail (most expensive but most reliable).
- Cumulative cost: ~$20. Budget remaining: ~$70.
- Holding v5; escalating with options per cron rules.
