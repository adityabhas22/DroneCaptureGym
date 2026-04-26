# Hackathon Submission Readiness Checklist

This checklist tracks the non-negotiable hackathon requirements against the current DroneCaptureGym repository state.

Status legend:

- `DONE` means the repo already satisfies the item based on local inspection or validation.
- `PARTIAL` means the repo has some support, but not enough for a strong submission.
- `MISSING` means the item is not present yet.
- `BLOCKED` means the item requires external credentials, deployment, or a real training run.

## Current Status

| Requirement | Status | Current Evidence | What Still Needs To Happen |
| --- | --- | --- | --- |
| Use latest OpenEnv and build on the framework | DONE | `pyproject.toml` uses `openenv-core>=0.2.3`; `openenv validate` passed with `[OK] DroneCaptureGym: Ready for multi-mode deployment`. | Keep dependency current before final submission. |
| Real-world task, not toy/game | DONE | README describes drone solar inspection; environment models high-level inspection director actions, evidence capture, safety, and reporting. | None for this requirement. |
| Typed OpenEnv models and server | DONE | `dronecaptureops/core/models.py`, `server/app.py`, and `openenv.yaml` exist and validate. | None for this requirement. |
| Dockerfile builds | BLOCKED | `Dockerfile` exists. Validator found it. | Docker daemon was not running locally, so build could not be verified. Start Docker Desktop and rerun validator. |
| HF Space is live and responds to `/reset` | BLOCKED | Local `/reset` ping passed using `http://127.0.0.1:8000`. | Push/deploy to Hugging Face Space and rerun validator with the real Space URL. |
| README links to HF Space | PARTIAL | README now has an explicit `TODO_HF_SPACE_URL` placeholder. | Replace placeholder with final public Space URL once deployed. |
| Root `inference.py` exists | DONE | `inference.py` exists and runs the three default tasks locally with scripted policy. | Confirm exact formatting against final evaluator. |
| Inference uses OpenAI-compatible client with env vars | PARTIAL | `inference.py` supports `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`, and falls back to scripted policy locally. | Confirm judges allow fallback when credentials are absent; otherwise make OpenAI-client path mandatory for submission mode. |
| Structured `[START]`, `[STEP]`, `[END]` logs | DONE | Local `inference.py --policy scripted` emits all three line types; `[END] score` now uses 2 decimals to match the OpsArena sample style. | Validate against final evaluator before submission. |
| Minimum 3 tasks with deterministic graders | DONE | Default inference tasks: `basic_thermal_survey`, `anomaly_confirmation`, `audit_grade_strict_grounding`. Reward/verifier stack is deterministic and scores in bounded range. | Consider documenting task difficulty more prominently in README. |
| Meaningful reward function with partial progress | DONE | README documents dense shaping, terminal scoring, safety/integrity gates, and reward components. Tests cover reward behavior. | None for this requirement. |
| Working training script using TRL/Unsloth/other RL framework | PARTIAL | `training/sft_warmstart.py` is a real TRL SFT script. `training/train_grpo.py` is only a scaffold, not a working RL trainer. | Implement or provide a working RL training script/notebook. Do not fake this. |
| Colab notebook so judges can rerun training | PARTIAL | `training/colab_training_template.md` provides a Colab-ready runbook. Notebook creation failed locally, so there is not yet an executed `.ipynb`. | Convert/run it as a real Colab notebook and link the executed notebook. |
| Evidence of actual training | MISSING | Plotting/tracking plumbing exists, but no real loss/reward plots or training artifacts are committed. | Run a real training job and save/link plots. Do not fabricate evidence. |
| Loss and reward plots from a real run | PARTIAL | `training/plot_training_metrics.py` can plot real logs from completed runs. | Generate plots from actual logged training/eval data. |
| Experimental tracking enabled | PARTIAL | `training/configs/sft_train_default.yaml` now reports to TensorBoard. | Run training and publish/link TensorBoard/W&B evidence. |
| Short writeup/video explaining env and training | PARTIAL | README now has `TODO_WRITEUP_OR_VIDEO_URL`. | Create a public writeup/video and replace the placeholder. |
| README motivates problem and explains env | PARTIAL | README explains the environment, action space, observation space, rewards, Docker, baseline tasks, checklist, and placeholders. | Add final results, Space link, training evidence, and public material links. |
| README links all additional materials | PARTIAL | README now lists all required placeholder URLs. | Replace all placeholders after deployment/training/writeup. |
| Avoid big video files in HF submission | DONE | No video files or large local media found in repo scan. | Keep media external via public URLs. |
| Pre-submission validator | PARTIAL | Provided validator passed Step 1 against local server; Step 2 failed due Docker daemon unavailable; manual `openenv validate` passed. | Start Docker and rerun full validator against real HF Space URL. |
| Client/server separation | DONE | `server/app.py` is a thin OpenEnv/FastAPI entrypoint; environment logic lives under `dronecaptureops/`. | Keep client scripts from importing server internals. |
| Standard Gym/OpenEnv API | DONE | `DroneCaptureOpsEnvironment` implements `reset`, `step`, and visible `state`. | Re-run `openenv validate` before final submission. |
| Reserved tool names avoided | DONE | Public tool registry uses inspection/flight/camera/report tool names, not `reset`, `step`, `state`, or `close`. | Re-check if any new tools are added. |
| Hidden verifier state protected | DONE | `tests/test_no_hidden_state_leakage.py` checks hidden defects and verifier-only fields are not exposed. | Keep this test passing if observations change. |
| Public story/video script | DONE | `docs/submission-video-script.md` provides a three-person 90-120 second narration, DroneKit/SITL demo guide, storyboard, and recording checklist. | Publish the final video/blog/slides externally and link it from README. |
| DroneKit demo honesty boundary | PARTIAL | Video guide explains that `GeometryController` is the training backend and `DroneKitSITLController` is currently a placeholder; DroneKit should be shown as a bridge/SITL demo unless fully implemented. | If recording DroneKit footage, avoid claiming the submitted Space directly controls DroneKit unless the adapter is implemented and validated. |
| Final placeholder sweep | MISSING | README and training runbook intentionally contain `TODO_*` placeholders. | Before submission, search for `TODO_HF_SPACE_URL`, `TODO_WRITEUP_OR_VIDEO_URL`, and other public-link placeholders and replace them. |
| License for public release | MISSING | README notes that no root `LICENSE` exists yet. | Add a license approved by the team before public release. |

## Items We Can Fix Without Running Real Training

These can be done locally without pretending that training happened:

- DONE: Align `inference.py` `[END] score` formatting to OpsArena sample expectations with 2-decimal scores.
- DONE: Update README with clearer task difficulty, baseline commands, and placeholders/TODOs for final public links.
- DONE: Add a dedicated submission section to README for HF Space, training notebook, plots, blog/video, and model artifact links.
- PARTIAL: Add a Colab-ready runbook at `training/colab_training_template.md`. Convert it to an executed Colab notebook before final submission.
- DONE: Enable TensorBoard experiment tracking config for future training runs. This only prepares tracking; it does not create evidence.
- DONE: Add `training/plot_training_metrics.py` to plot training/eval metrics once real logs exist.
- DONE: Add `scripts/validate-submission.sh` copy of the validator you provided, so future checks are reproducible from the repo.
- DONE: Add README warnings that final plots and links must come from a real run, not generated placeholders.
- DONE: Correct checklist/README references to the actual OpenEnv server entrypoint, `server/app.py`.
- DONE: Add a submission video script/storyboard that can be recorded without committing large media files.
- PENDING: Add a root `LICENSE` before the repo is made public.
- PENDING: Run a final placeholder sweep before submission so the README does not ship with `TODO_*` public links.
- PENDING: Rerun tests and `openenv validate` after these documentation/config fixes.

## Items We Should Not Fake Or Do Without Your Approval

- Real RL training run.
- Loss/reward plots claiming to be from training.
- Public HF Space deployment under your account.
- Public blog/video/upload links.
- Pushing trained model artifacts to Hugging Face.
- Any paid HF Jobs / GPU run.
- Choosing a public release license without team approval.

## Recommended Next Order

1. Run local checks after the low-risk compliance fixes: inference score formatting, README submission links section, tracking config, validator script.
2. Start Docker Desktop and rerun the provided validator locally.
3. Deploy the environment to a Hugging Face Space and rerun validator with the real Space URL.
4. Record or publish the short writeup/video using the prepared story script.
5. Add a team-approved root `LICENSE`.
6. Collect real training evidence from the training workstream without fabricating plots or metrics.
7. Update README with final HF Space, plots, model artifacts, notebook, and writeup/video links.
8. Run a final `TODO_` placeholder sweep and rerun the full release gate.

