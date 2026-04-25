# Engineering Backlog

These are repository and platform implementation tasks, separate from RL mission tasks for the LLM agent.

1. OpenEnv deployment readiness
   - Add the exact `server = "server.app:main"` script entry expected by OpenEnv local validation.
   - Generate `uv.lock` so `openenv validate .` passes locally.

2. Typed tool argument validation
   - Replace loose dictionary argument handling with stronger per-tool schemas.
   - Improve invalid argument messages and add tests for wrong types and unsafe ranges.

3. Trajectory benchmark suite
   - Add weak, random, scripted, and strong baseline rollouts across fixed seeds.
   - Produce score summaries and regression tests.

4. Capture evidence improvements
   - Make RGB close-up pairing target-specific.
   - Improve standoff, angle, and quality scoring.
   - Require anomaly evidence to actually see the relevant asset.

5. Reward hardening
   - Reduce reward gaming through stricter report citations, modality checks, quality thresholds, return-home compliance, and redundancy penalties.

6. Solar scenario variants
   - Add multiple deterministic solar layouts, no-fly configurations, weather bands, defect types, and held-out evaluation seeds.

7. Real domain expansion
   - Implement actual bridge, construction, and industrial scenario builders instead of placeholder solar aliases.

8. Training readiness
   - Turn `training/train_grpo.py` into a real OpenEnv/TRL rollout wrapper once prompts, action formatting, and eval splits are defined.

9. Episode observability
   - Add richer episode traces: action timeline, capture table, reward evolution, safety events, and final report diagnostics.

10. Safety extensions
    - Add privacy-zone capture checks, return-battery feasibility, dynamic obstacles, and stronger tests for unsafe plans.
