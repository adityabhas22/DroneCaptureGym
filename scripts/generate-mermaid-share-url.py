"""Generate Mermaid Live Editor and Mermaid Ink URLs for a .mmd file."""

from __future__ import annotations

import base64
import json
import sys
import zlib
from pathlib import Path


def encode_mermaid(code: str) -> str:
    state = {
        "code": code,
        "mermaid": {"theme": "default"},
        "autoSync": True,
        "updateDiagram": True,
    }
    payload = json.dumps(state, separators=(",", ":")).encode("utf-8")
    compressed = zlib.compress(payload, level=9)
    return base64.urlsafe_b64encode(compressed).decode("ascii").rstrip("=")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python scripts/generate-mermaid-share-url.py docs/submission-hero-flow.mmd", file=sys.stderr)
        return 2

    diagram_path = Path(sys.argv[1])
    code = diagram_path.read_text(encoding="utf-8")
    encoded = encode_mermaid(code)

    print("Mermaid Live Editor URL:")
    print(f"https://mermaid.live/edit#pako:{encoded}")
    print()
    print("Direct SVG render URL:")
    print(f"https://mermaid.ink/svg/pako:{encoded}")
    print()
    print(f"Encoded URL payload length: {len(encoded)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
