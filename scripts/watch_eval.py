#!/usr/bin/env python3
"""Live scoreboard for the qwen3_32b_short eval (or any eval_models.py run).

Reads the per-cell JSONL (which streams as cells finish) plus the tail of
the log to show what's currently in flight. Re-run anytime; it just snapshots.

Usage:
    python scripts/watch_eval.py                                       # default 32b run
    python scripts/watch_eval.py --jsonl artifacts/eval/qwen3_4b_short.jsonl
    watch -n 5 -c python scripts/watch_eval.py                         # live update
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", default="artifacts/eval/qwen3_32b_short.jsonl")
    p.add_argument("--log", default=None, help="defaults to <jsonl>.log")
    p.add_argument("--pid", default=None, help="defaults to <jsonl>.pid")
    p.add_argument("--total", type=int, default=30)
    args = p.parse_args()

    jsonl = Path(args.jsonl)
    stem = jsonl.with_suffix("")
    log = Path(args.log) if args.log else stem.with_suffix(".log")
    pidfile = Path(args.pid) if args.pid else stem.with_suffix(".pid")

    cells = []
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            line = line.strip()
            if line:
                cells.append(json.loads(line))

    # State
    state = "UNKNOWN"
    if pidfile.exists():
        try:
            pid = int(pidfile.read_text().strip())
            os.kill(pid, 0)
            state = f"RUNNING pid={pid}"
        except (ProcessLookupError, ValueError):
            state = "DONE"

    n = len(cells)
    print(f"== {jsonl.name} :: {state} :: {n}/{args.total} cells ==")

    if not cells:
        print("(no cells finished yet)")
        if log.exists():
            print("\n--- last log lines ---")
            print(subprocess.run(["tail", "-8", str(log)], capture_output=True, text=True).stdout)
        return 0

    # Per-cell board
    print(f"\n{'#':<3} {'task_id':<32} {'seed':<5} {'rew':<6} {'steps':<6} {'parse_err':<10} {'failure_mode':<22} top_tools")
    for i, c in enumerate(cells, 1):
        tc = c.get("tool_calls") or {}
        top = sorted([(k, v) for k, v in tc.items() if not k.startswith("_")], key=lambda x: -x[1])[:4]
        top_s = " ".join(f"{k}×{v}" for k, v in top)
        print(
            f"{i:<3} {c['task_id']:<32} {c['seed']:<5} {c['total_reward']:<6.3f} "
            f"{c['steps']:<6} {c['parse_error_count']:<10} {(c.get('failure_mode') or '?'):<22} {top_s}"
        )

    # Aggregates
    success = sum(1 for c in cells if c["success"])
    complete = sum(1 for c in cells if c.get("complete"))
    mean_reward = sum(c["total_reward"] for c in cells) / n
    mean_steps = sum(c["steps"] for c in cells) / n
    parse_total = sum(c["parse_error_count"] for c in cells)
    fm = Counter(c.get("failure_mode") or "?" for c in cells)
    cp = Counter()
    for c in cells:
        for k, v in (c.get("checkpoints") or {}).items():
            if v:
                cp[k] += 1

    print(f"\n-- aggregates --")
    print(f"success: {success}/{n} ({100*success/n:.1f}%)   complete: {complete}/{n}   mean_reward: {mean_reward:.3f}   mean_steps: {mean_steps:.1f}   parse_errors_total: {parse_total}")
    print(f"failure modes: {dict(fm.most_common())}")
    print(f"checkpoint hits: {dict(cp.most_common())}")

    # Tool call totals across all completed cells
    tool_total = Counter()
    for c in cells:
        for k, v in (c.get("tool_calls") or {}).items():
            tool_total[k] += v
    print(f"\n-- tool-call totals across {n} cells --")
    for tool, cnt in tool_total.most_common():
        marker = "  ⚠" if tool.startswith("_") else "   "
        print(f" {marker} {tool:<28} {cnt}")

    # In-flight cell from log
    if log.exists():
        log_tail = subprocess.run(
            ["grep", "-E", "cell [0-9]+/[0-9]+|HTTP/1.1 (4|5)", str(log)],
            capture_output=True, text=True,
        ).stdout.splitlines()
        recent = [ln for ln in log_tail[-10:]]
        if recent:
            print(f"\n-- recent log signal --")
            for ln in recent:
                print(" ", ln[:200])

    return 0


if __name__ == "__main__":
    sys.exit(main())
