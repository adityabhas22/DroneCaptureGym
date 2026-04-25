#!/usr/bin/env python3
"""Forensics over the per-turn audit JSONL produced by training/eval_models.py.

Usage examples:
    # Summary across all turns: counts, failure modes, truncations
    python scripts/inspect_audit.py artifacts/eval/qwen3_32b_short.audit.jsonl

    # Show every parse_error with truncated response_text + thinking
    python scripts/inspect_audit.py artifacts/eval/qwen3_32b_short.audit.jsonl --failures

    # Drill into one cell's turns chronologically
    python scripts/inspect_audit.py artifacts/eval/qwen3_32b_short.audit.jsonl --task basic_thermal_survey --seed 1

    # Just the cells that crashed mid-rollout
    python scripts/inspect_audit.py artifacts/eval/qwen3_32b_short.audit.jsonl --crashes
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("audit_path", type=Path)
    p.add_argument("--failures", action="store_true", help="Show every turn with parse_error or api_error.")
    p.add_argument("--crashes", action="store_true", help="Show synthetic crash markers (cell aborted mid-rollout).")
    p.add_argument("--task", help="Filter to one task_id.")
    p.add_argument("--seed", type=int, help="Filter to one seed.")
    p.add_argument("--max", type=int, default=20, help="Max turn snippets to print in detail mode.")
    p.add_argument("--text-chars", type=int, default=600, help="Truncation length for response_text/thinking dumps.")
    args = p.parse_args()

    if not args.audit_path.exists():
        raise SystemExit(f"missing: {args.audit_path}")

    rows = [json.loads(line) for line in args.audit_path.read_text().splitlines() if line.strip()]
    if args.task:
        rows = [r for r in rows if r.get("task_id") == args.task]
    if args.seed is not None:
        rows = [r for r in rows if r.get("seed") == args.seed]

    crash_rows = [r for r in rows if r.get("crash")]
    turn_rows = [r for r in rows if not r.get("crash")]

    if args.crashes:
        print(f"== {len(crash_rows)} crashed cells ==\n")
        for r in crash_rows:
            print(f"  {r['task_id']:<32s} seed={r['seed']:<3} after {r.get('n_turns_before_crash')} turns "
                  f"-- {r.get('crash_exc_type')}: {r.get('crash_message', '')[:200]}")
        return 0

    if args.failures:
        bad = [r for r in turn_rows if r.get("parse_error") or r.get("api_error")]
        print(f"== {len(bad)}/{len(turn_rows)} turns with errors ==\n")
        for r in bad[: args.max]:
            print(f"--- {r['task_id']} seed={r['seed']} step={r['step']} ---")
            print(f"  parse_error    : {r.get('parse_error')}")
            print(f"  finish_reason  : {r.get('finish_reason')}")
            print(f"  truncated      : {r.get('truncated')}")
            print(f"  retries        : {r.get('retries')}")
            print(f"  tokens p/c/t   : {r.get('prompt_tokens')}/{r.get('completion_tokens')}/{r.get('total_tokens')}")
            api_err = r.get("api_error") or {}
            if api_err:
                print(f"  api_error.code : {api_err.get('code')} status={api_err.get('status_code')}")
                fg = api_err.get("failed_generation") or ""
                if fg:
                    print(f"  failed_generation (first {args.text_chars}):")
                    print(f"    {fg[:args.text_chars]!r}")
            text = r.get("response_text") or ""
            if text:
                print(f"  response_text (first {args.text_chars}):")
                print(f"    {text[:args.text_chars]!r}")
            think = r.get("thinking_content") or ""
            if think:
                print(f"  thinking_content (first {args.text_chars}):")
                print(f"    {think[:args.text_chars]!r}")
            print()
        if len(bad) > args.max:
            print(f"... {len(bad) - args.max} more (use --max).")
        return 0

    # Default: aggregate summary.
    n = len(turn_rows)
    if n == 0:
        print("(empty audit)")
        return 0
    parse_errs = sum(1 for r in turn_rows if r.get("parse_error"))
    api_errs = sum(1 for r in turn_rows if r.get("api_error"))
    truncated = sum(1 for r in turn_rows if r.get("truncated"))
    finish = Counter(r.get("finish_reason") for r in turn_rows)
    by_task = Counter((r["task_id"], r["seed"]) for r in turn_rows)
    completion = [r.get("completion_tokens") for r in turn_rows if r.get("completion_tokens")]
    completion_max = max(completion) if completion else 0
    completion_mean = sum(completion) / len(completion) if completion else 0
    thinking_chars = [len(r.get("thinking_content") or "") for r in turn_rows]
    thinking_max = max(thinking_chars) if thinking_chars else 0

    print(f"== audit summary :: {args.audit_path.name} ==")
    print(f"turns: {n}   parse_errors: {parse_errs}   api_errors: {api_errs}   truncated(finish=length): {truncated}")
    print(f"finish_reasons: {dict(finish)}")
    print(f"completion_tokens: max={completion_max}  mean={completion_mean:.0f}")
    print(f"thinking_content chars (max in any single turn): {thinking_max}")
    print(f"cells covered: {len(by_task)}   (turns/cell range: {min(by_task.values()) if by_task else 0}–{max(by_task.values()) if by_task else 0})")
    if crash_rows:
        print(f"\n!! {len(crash_rows)} crashed cells (use --crashes for detail)")
    if parse_errs or api_errs:
        print(f"\nuse --failures to inspect bad turns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
