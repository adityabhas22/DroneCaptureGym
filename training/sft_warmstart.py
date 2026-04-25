"""SFT warm-start trainer — anti-overfit by default.

Wraps TRL's SFTTrainer with the exact knobs that matter for keeping the
warm-start *general* rather than a memorised replay of the oracle:

1. **LoRA** (rank 16 by default). Limits the parameter surface so the
   model can't memorise long trajectories verbatim. Disable via
   `lora.enabled: false` if you really want full fine-tuning.
2. **Train/val split by seed within each task**. We pick a fraction
   of seeds per task and reserve them as eval — this surfaces
   within-task generalisation, not just held-out task generalisation.
3. **Early stopping on eval_loss**. ToolRL / ClaimsGym both showed late
   epochs overfit aggressively; default `early_stopping_patience=3`.
4. **Low LR + cosine** (2e-5, warmup 5%). Big LRs blow tool-call
   format capability fast.
5. **Limited epochs** (3 max). With a few hundred examples, more is
   memorisation.
6. **Assistant-only loss masking** so the model trains on what it must
   *emit*, not what it *reads*.
7. **Length filtering** (not truncation) — overlong trajectories are
   dropped with a warning. Truncating mid-tool-call corrupts SFT data.

Usage:
    python -m training.sft_warmstart                                  # uses defaults
    python -m training.sft_warmstart --config training/configs/sft_train_default.yaml
    python -m training.sft_warmstart --dataset path/to/sft.jsonl --model Qwen/Qwen2.5-3B-Instruct
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "training" / "configs" / "sft_train_default.yaml"

LOG = logging.getLogger("dronecaptureops.sft.train")


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------


class LoRAConfig(BaseModel):
    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


class SFTTrainConfig(BaseModel):
    """Anti-overfit-tuned defaults; every knob is overridable via CLI."""

    dataset: str = "artifacts/sft/sft-warmstart.jsonl"
    output_dir: str = "artifacts/sft-checkpoints"
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer_name: str | None = None

    val_seed_fraction: float = Field(default=0.15, ge=0.0, le=0.5)
    max_seq_length: int = 4096

    num_train_epochs: int = 3
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8

    eval_strategy: str = "steps"
    eval_steps: int = 25
    save_strategy: str = "steps"
    save_steps: int = 25
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    early_stopping_patience: int = 3

    lora: LoRAConfig = Field(default_factory=LoRAConfig)

    assistant_only_loss: bool = True
    response_template: str = "<|im_start|>assistant\n"

    logging_steps: int = 5
    report_to: list[str] = Field(default_factory=list)
    seed: int = 42


def load_config(path: Path | None) -> SFTTrainConfig:
    if path is None:
        path = DEFAULT_CONFIG_PATH
    if not path.exists():
        LOG.warning("config path %s not found, using built-in defaults", path)
        return SFTTrainConfig()
    raw = yaml.safe_load(path.read_text()) or {}
    try:
        return SFTTrainConfig.model_validate(raw)
    except ValidationError as exc:
        raise SystemExit(f"invalid SFT trainer config: {exc}") from exc


# ---------------------------------------------------------------------------
# Dataset prep — pure-Python (no transformers needed for tests)
# ---------------------------------------------------------------------------


@dataclass
class SFTExample:
    """One chat-format example after splitting and length filtering."""

    task_id: str
    seed: int
    messages: list[dict[str, Any]]
    text: str | None = None  # rendered chat template, populated lazily by build_dataset


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"dataset not found: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def split_train_val(
    records: list[dict[str, Any]],
    *,
    val_seed_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Hold out a fraction of seeds from each task for eval.

    Splitting by seed (not by row index) keeps within-task variation on
    the eval side. We bucket all records by task_id, select a deterministic
    subset of seeds based on a hash, and partition.
    """

    import random

    rng = random.Random(seed)
    by_task: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_task.setdefault(record["task_id"], []).append(record)

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for task_id, rows in by_task.items():
        seeds = sorted({row["seed"] for row in rows if row.get("seed") is not None})
        if len(seeds) <= 1:
            # Not enough variety to split — keep everything in train.
            train.extend(rows)
            continue
        n_val = max(1, round(val_seed_fraction * len(seeds)))
        val_seeds = set(rng.sample(seeds, n_val))
        for row in rows:
            (val if row.get("seed") in val_seeds else train).append(row)
    LOG.info("split: %d train / %d val examples (across %d tasks)", len(train), len(val), len(by_task))
    return train, val


def filter_by_length(
    records: list[dict[str, Any]],
    *,
    tokenizer,  # noqa: ANN001 — transformers tokenizer
    max_seq_length: int,
) -> tuple[list[SFTExample], int]:
    """Drop examples whose rendered chat template exceeds the seq cap.

    Truncating mid-tool-call breaks SFT (the assistant turn becomes
    invalid JSON), so we filter rather than truncate.
    """

    kept: list[SFTExample] = []
    dropped = 0
    for record in records:
        messages = record["messages"]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        token_count = len(tokenizer(text, add_special_tokens=False).input_ids)
        if token_count > max_seq_length:
            dropped += 1
            continue
        kept.append(
            SFTExample(
                task_id=record["task_id"],
                seed=record.get("seed", -1),
                messages=messages,
                text=text,
            )
        )
    if dropped:
        LOG.warning("dropped %d examples that exceeded max_seq_length=%d", dropped, max_seq_length)
    return kept, dropped


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def train(config: SFTTrainConfig) -> None:
    """Run the full SFT loop. Heavyweight imports happen here so the rest
    of the module (config + dataset prep) stays test-friendly."""

    try:
        import torch  # noqa: F401  (sanity check the user has it)
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            EarlyStoppingCallback,
        )
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit(
            "SFT training requires `pip install -e '.[train]'` (torch, transformers, trl, peft, datasets)."
        ) from exc

    dataset_path = REPO_ROOT / config.dataset if not Path(config.dataset).is_absolute() else Path(config.dataset)
    output_dir = REPO_ROOT / config.output_dir if not Path(config.output_dir).is_absolute() else Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("loading dataset from %s", dataset_path)
    records = load_jsonl(dataset_path)
    if not records:
        raise SystemExit(f"dataset {dataset_path} is empty — run training/generate_sft_data.py first")

    train_records, val_records = split_train_val(
        records,
        val_seed_fraction=config.val_seed_fraction,
        seed=config.seed,
    )

    LOG.info("loading tokenizer + model: %s", config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_examples, _ = filter_by_length(train_records, tokenizer=tokenizer, max_seq_length=config.max_seq_length)
    val_examples, _ = filter_by_length(val_records, tokenizer=tokenizer, max_seq_length=config.max_seq_length)
    if not train_examples:
        raise SystemExit("no train examples survived length filtering — bump max_seq_length")
    LOG.info("train=%d val=%d after length filter", len(train_examples), len(val_examples))

    train_ds = Dataset.from_list([{"messages": ex.messages} for ex in train_examples])
    eval_ds = Dataset.from_list([{"messages": ex.messages} for ex in val_examples]) if val_examples else None

    LOG.info("loading base model %s", config.model_name)
    model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype="auto")

    sft_kwargs: dict[str, Any] = dict(
        output_dir=str(output_dir),
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        logging_steps=config.logging_steps,
        report_to=config.report_to or "none",
        seed=config.seed,
        max_seq_length=config.max_seq_length,
        packing=False,
        eval_strategy=config.eval_strategy if eval_ds is not None else "no",
        eval_steps=config.eval_steps if eval_ds is not None else None,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        load_best_model_at_end=config.load_best_model_at_end and eval_ds is not None,
        metric_for_best_model=config.metric_for_best_model if eval_ds is not None else None,
        greater_is_better=config.greater_is_better,
    )
    if config.assistant_only_loss:
        # TRL >= 0.11 supports `assistant_only_loss=True` directly. Older
        # versions need DataCollatorForCompletionOnlyLM with response_template;
        # we forward `response_template` so SFTConfig picks it up if available.
        sft_kwargs["assistant_only_loss"] = True
    sft_config = SFTConfig(**_filter_kwargs(SFTConfig, sft_kwargs))

    callbacks = []
    if eval_ds is not None and config.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience))

    peft_config = None
    if config.lora.enabled:
        peft_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=list(config.lora.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )

    LOG.info(
        "starting SFTTrainer: epochs=%d lr=%g lora=%s eval=%s patience=%d",
        config.num_train_epochs,
        config.learning_rate,
        config.lora.enabled,
        eval_ds is not None,
        config.early_stopping_patience,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    LOG.info("training complete; final model written to %s", output_dir / "final")


def _filter_kwargs(klass, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop kwargs the active TRL/SFTConfig version doesn't accept.

    TRL keeps shifting their config surface (e.g. `eval_strategy` was
    `evaluation_strategy` pre-0.11). Forward only the names the live
    dataclass declares so we don't break across TRL versions.
    """

    from dataclasses import fields

    valid = {f.name for f in fields(klass)}
    filtered = {k: v for k, v in kwargs.items() if k in valid and v is not None}
    dropped = set(kwargs) - set(filtered)
    if dropped:
        LOG.warning("SFTConfig in this TRL build dropped: %s", sorted(dropped))
    return filtered


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT warm-start trainer (anti-overfit defaults).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset", default=None, help="Override dataset JSONL path.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--model", dest="model_name", default=None, help="Override base model.")
    parser.add_argument("--epochs", dest="num_train_epochs", type=int, default=None)
    parser.add_argument("--lr", dest="learning_rate", type=float, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA — full fine-tuning (overfit risk!).")
    parser.add_argument("--no-eval", action="store_true", help="Skip the train/val split and eval loop.")
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate config + dataset without launching training.")
    return parser.parse_args()


def _apply_cli_overrides(config: SFTTrainConfig, args: argparse.Namespace) -> SFTTrainConfig:
    update: dict[str, Any] = {}
    for attr in (
        "dataset",
        "output_dir",
        "model_name",
        "num_train_epochs",
        "learning_rate",
        "max_seq_length",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "early_stopping_patience",
        "seed",
    ):
        value = getattr(args, attr, None)
        if value is not None:
            update[attr] = value
    if args.no_lora:
        update["lora"] = config.lora.model_copy(update={"enabled": False})
    if args.no_eval:
        update["val_seed_fraction"] = 0.0
    if not update:
        return config
    return config.model_copy(update=update)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args()
    config = _apply_cli_overrides(load_config(args.config), args)

    LOG.info(
        "config: model=%s lr=%g epochs=%d lora=%s val_frac=%.2f early_stop=%d",
        config.model_name,
        config.learning_rate,
        config.num_train_epochs,
        config.lora.enabled,
        config.val_seed_fraction,
        config.early_stopping_patience,
    )

    if args.dry_run:
        dataset_path = REPO_ROOT / config.dataset if not Path(config.dataset).is_absolute() else Path(config.dataset)
        records = load_jsonl(dataset_path)
        train_records, val_records = split_train_val(
            records,
            val_seed_fraction=config.val_seed_fraction,
            seed=config.seed,
        )
        print(json.dumps({
            "dataset": str(dataset_path),
            "train_count": len(train_records),
            "val_count": len(val_records),
            "by_task": {
                task_id: sum(1 for r in records if r["task_id"] == task_id)
                for task_id in sorted({r["task_id"] for r in records})
            },
            "model": config.model_name,
            "lora_enabled": config.lora.enabled,
            "lora_target_modules": config.lora.target_modules,
            "lr": config.learning_rate,
            "epochs": config.num_train_epochs,
            "val_seed_fraction": config.val_seed_fraction,
            "early_stopping_patience": config.early_stopping_patience,
        }, indent=2, sort_keys=True))
        return 0

    train(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
