#!/usr/bin/env python3
"""Tail logs for a running HF Jobs run.

Usage:
    python scripts/tail_job.py <job_id>
    # or, with a default token from .env / HF_AUTH_TOKEN / HF_TOKEN

The job keeps running if you Ctrl+C — this script just detaches your
local terminal from the log stream.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _resolve_token() -> str:
    for name in ("HF_TOKEN", "HF_AUTH_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value
    # Fall back to .env in repo root, lazily.
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and any(line.startswith(f"{n}=") for n in ("HF_TOKEN", "HF_AUTH_TOKEN")):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("no HF token in env or .env")


def main() -> int:
    parser = argparse.ArgumentParser(description="Tail HF Jobs logs.")
    parser.add_argument("job_id")
    parser.add_argument("--namespace", default=None, help="Override namespace (defaults to your account).")
    parser.add_argument("--no-follow", action="store_true", help="Print current logs and exit.")
    args = parser.parse_args()

    from huggingface_hub import fetch_job_logs, inspect_job

    token = _resolve_token()
    info = inspect_job(job_id=args.job_id, token=token, namespace=args.namespace)
    status = getattr(info, "status", info)
    print(f"# job {args.job_id} status: {status}", file=sys.stderr)

    for line in fetch_job_logs(
        job_id=args.job_id,
        token=token,
        namespace=args.namespace,
        follow=not args.no_follow,
    ):
        sys.stdout.write(line if line.endswith("\n") else line + "\n")
        sys.stdout.flush()

    final = inspect_job(job_id=args.job_id, token=token, namespace=args.namespace)
    print(f"\n# final status: {getattr(final, 'status', final)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
