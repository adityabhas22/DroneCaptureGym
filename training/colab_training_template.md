# DroneCaptureOps Colab Training Template

This is a Colab-ready runbook. Convert it to a notebook or paste the cells into Colab before final submission.

Important: this file is not training evidence. Final README links must point to a real executed notebook, real logs, and real loss/reward plots.

## 1. Clone And Install

```bash
git clone <PUBLIC_REPO_URL> DroneCaptureGym
cd DroneCaptureGym
pip install -e ".[dev,train]"
```

Placeholder to fill later:

- `<PUBLIC_REPO_URL>`: public GitHub or Hugging Face repo URL.

## 2. Validate OpenEnv

```bash
openenv validate
python inference.py --policy scripted
```

Expected before training:

- `openenv validate` should print an `[OK]` line.
- `inference.py` should run the three default tasks and print `[START]`, `[STEP]`, and `[END]` records.

## 3. Generate Warm-Start Data

```bash
python -m training.generate_sft_data \
  --config training/configs/sft_default.yaml \
  --output artifacts/sft/sft-warmstart.jsonl
```

This creates supervised warm-start trajectories from deterministic policies. It is useful before RL, but it is not itself an RL run.

## 4. Run A Tiny Smoke Training Job

Use a very small model for a Colab smoke test:

```bash
python -m training.sft_warmstart \
  --config training/configs/sft_train_default.yaml \
  --dataset artifacts/sft/sft-warmstart.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir artifacts/sft-checkpoints-smoke \
  --epochs 1 \
  --max-seq-length 2048 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 1
```

For the final run, use the configured model in `training/configs/sft_train_default.yaml` or a stronger approved base model.

## 5. Enable And View Tracking

The default training config reports to TensorBoard.

```bash
tensorboard --logdir artifacts/sft-checkpoints
```

If using W&B instead, configure `WANDB_API_KEY` and change `report_to` in `training/configs/sft_train_default.yaml` to:

```yaml
report_to:
  - wandb
```

## 6. Plot Real Metrics

After a completed run:

```bash
python -m training.plot_training_metrics \
  --trainer-log artifacts/sft-checkpoints/metrics/trainer_log_history.json \
  --output-dir artifacts/submission-plots
```

If you have eval JSONL from `training/eval_models.py`, include it:

```bash
python -m training.plot_training_metrics \
  --trainer-log artifacts/sft-checkpoints/metrics/trainer_log_history.json \
  --eval-jsonl artifacts/eval/model-eval.jsonl \
  --output-dir artifacts/submission-plots
```

Expected real output files:

- `artifacts/submission-plots/loss_curve.png`
- `artifacts/submission-plots/reward_curve.png` if eval rewards are provided

## 7. Final Materials To Link In README

Fill these after real execution:

- Executed Colab notebook URL: `TODO`
- TensorBoard or W&B run URL: `TODO`
- Loss plot URL: `TODO`
- Reward plot URL: `TODO`
- Trained adapter/model URL: `TODO`
- HF Space URL: `TODO`
- Short blog/video URL: `TODO`

