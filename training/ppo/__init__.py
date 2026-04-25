"""PPO trainer modules for DroneCaptureOps.

Single-process, single-GPU PPO over the existing RolloutRunner +
RewardAggregator + VLLMPolicy stack. The math (GAE, clipped policy loss,
critic loss, KL estimators) is adapted with attribution from
https://github.com/NovaSky-AI/SkyRL and (transitively)
https://github.com/volcengine/verl — both Apache-2.0.

Module map:
    loss.py             — masked GAE, PPO clipped loss, critic loss, KL
    reward_placement.py — token-level reward placement at turn-end tokens
    tokenization.py     — chat-template → tokens with assistant mask and
                           turn-boundary metadata
    rollout_pool.py     — concurrent rollouts (threads sharing one vLLM)
    trainer.py          — the main PPO loop
"""
