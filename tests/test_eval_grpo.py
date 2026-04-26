"""Unit tests for ``training/eval_grpo.py``.

We unit-test two contracts that are easy to break without realizing it:

1. ``parse_variant`` covers every CLI shape we document in --help
   (``base``, ``name=adapter``, ``name:repo:subfolder``).
2. ``build_comparison_markdown`` produces a valid markdown table that
   includes every variant + every task that any variant evaluated, so
   downstream consumers (the comparison.md uploaded to the output repo)
   never silently drop a model.

The end-to-end ``evaluate_variant`` path is exercised by the actual eval
job — it requires loading a 4B HF model so it cannot run in unit tests.
"""

from __future__ import annotations

import pytest

from training.eval_grpo import (
    DEFAULT_HELD_OUT,
    VariantSpec,
    build_comparison_markdown,
    parse_variant,
)


# ---------------------------------------------------------------------------
# parse_variant
# ---------------------------------------------------------------------------


def test_parse_variant_base_alone() -> None:
    spec = parse_variant("base")
    assert spec == VariantSpec(name="base", adapter_spec=None)


def test_parse_variant_equals_form_with_adapter() -> None:
    spec = parse_variant("sft=adityabhaskara/dronecaptureops-sft-qwen3-4b:last-checkpoint")
    assert spec.name == "sft"
    assert spec.adapter_spec == "adityabhaskara/dronecaptureops-sft-qwen3-4b:last-checkpoint"


def test_parse_variant_equals_form_base_means_no_adapter() -> None:
    spec = parse_variant("base=base")
    assert spec == VariantSpec(name="base", adapter_spec=None)


def test_parse_variant_colon_form_repo_only() -> None:
    spec = parse_variant("sft:adityabhaskara/dronecaptureops-sft-qwen3-4b")
    assert spec.name == "sft"
    assert spec.adapter_spec == "adityabhaskara/dronecaptureops-sft-qwen3-4b"


def test_parse_variant_colon_form_repo_plus_subfolder() -> None:
    spec = parse_variant("grpo:rachit-suresh/dronecaptureops-grpo-fixspan:output/final_10/adapter")
    assert spec.name == "grpo"
    assert spec.adapter_spec == "rachit-suresh/dronecaptureops-grpo-fixspan:output/final_10/adapter"


def test_parse_variant_strips_whitespace() -> None:
    spec = parse_variant("  base  ")
    assert spec.name == "base"
    assert spec.adapter_spec is None


# ---------------------------------------------------------------------------
# build_comparison_markdown
# ---------------------------------------------------------------------------


def _stub_summary(name: str, adapter: str | None, *, success_rate: float, mean_reward: float) -> dict:
    """Minimum keys ``build_comparison_markdown`` reads from a per-variant summary."""

    return {
        "variant": name,
        "adapter_spec": adapter,
        "base_model": "Qwen/Qwen3-4B-Instruct-2507",
        "n": 14,
        "success_rate": success_rate,
        "mean_total_reward": mean_reward,
        "mean_process_reward": mean_reward * 0.4,
        "mean_steps": 12.5,
        "parse_error_rate": 0.05,
        "elapsed_secs": 123.4,
        "seeds_per_task": 2,
        "max_episode_steps": 20,
        "by_task": {
            "edge_row_quality_bar": {
                "n": 2,
                "success_rate": success_rate,
                "mean_total_reward": mean_reward,
                "mean_process_reward": mean_reward * 0.4,
                "mean_steps": 12.5,
                "mean_parse_errors": 0.5,
            },
            "honest_partial_report_open_items": {
                "n": 2,
                "success_rate": success_rate * 0.5,
                "mean_total_reward": mean_reward * 0.7,
                "mean_process_reward": mean_reward * 0.3,
                "mean_steps": 14.0,
                "mean_parse_errors": 0.0,
            },
        },
        "failure_modes": {
            "no_capture": 0.6 if name != "grpo" else 0.4,
            "no_takeoff": 0.1,
            "format_collapse": 0.05,
        },
    }


def test_build_comparison_markdown_includes_every_variant_in_headline() -> None:
    md = build_comparison_markdown([
        _stub_summary("base", None, success_rate=0.0, mean_reward=0.01),
        _stub_summary("sft", "adityabhaskara/...", success_rate=0.05, mean_reward=0.04),
        _stub_summary("grpo", "rachit-suresh/...", success_rate=0.10, mean_reward=0.06),
    ])
    # Headline rows
    assert "| base |" in md
    assert "| sft |" in md
    assert "| grpo |" in md
    # Section headings
    assert "## Headline" in md
    assert "## Per-task mean reward" in md
    assert "## Per-task success rate" in md
    assert "## Failure-mode distribution" in md


def test_build_comparison_markdown_per_task_includes_union_of_task_ids() -> None:
    base = _stub_summary("base", None, success_rate=0.0, mean_reward=0.01)
    grpo = _stub_summary("grpo", "rachit-suresh/...", success_rate=0.1, mean_reward=0.06)
    # GRPO has a task that base did not eval on.
    grpo["by_task"]["return_margin_decision_point"] = {
        "n": 2,
        "success_rate": 0.5,
        "mean_total_reward": 0.2,
        "mean_process_reward": 0.05,
        "mean_steps": 18.0,
        "mean_parse_errors": 0.0,
    }
    md = build_comparison_markdown([base, grpo])
    # All three tasks present; missing cell rendered with em-dash.
    assert "edge_row_quality_bar" in md
    assert "honest_partial_report_open_items" in md
    assert "return_margin_decision_point" in md
    assert "—" in md, "missing-cell em-dash placeholder should appear in per-task tables"


def test_build_comparison_markdown_handles_empty_input() -> None:
    md = build_comparison_markdown([])
    assert "No variants completed" in md


def test_default_held_out_is_seven_distinct_tasks() -> None:
    assert len(DEFAULT_HELD_OUT) == 7
    assert len(set(DEFAULT_HELD_OUT)) == 7
