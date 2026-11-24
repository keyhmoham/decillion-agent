#!/usr/bin/env python3
"""Download the Agents-A1-4B tokenizer (Apache-2.0, InternScience/Agents-A1-4B) with checksum pinning.

The backend uses it for exact prompt-token accounting (context compaction decisions). Without it the
backend falls back to a conservative character-based estimate, so a failed download is not fatal.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://huggingface.co/InternScience/Agents-A1-4B/resolve/main/tokenizer.json"
SHA256 = "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4"
DEST = Path(sys.argv[1] if len(sys.argv) > 1 else "app/llm/assets/tokenizer.json")


def main() -> int:
    if DEST.exists() and hashlib.sha256(DEST.read_bytes()).hexdigest() == SHA256:
        print(f"tokenizer already present: {DEST}")
        return 0
    DEST.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(URL, timeout=120) as r:  # noqa: S310
            data = r.read()
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: could not download tokenizer ({exc}); falling back to estimated token counts", file=sys.stderr)
        return 0
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256:
        print(f"WARNING: tokenizer checksum mismatch ({digest}); not installing it", file=sys.stderr)
        return 0
    DEST.write_bytes(data)
    print(f"tokenizer installed: {DEST} ({len(data)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
