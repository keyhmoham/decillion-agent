#!/usr/bin/env python3
"""End-to-end smoke run against a *running* backend (REST only).

    python scripts/e2e.py --url http://localhost:8000 --mode agent \
        "Create fib.py that prints the first 10 Fibonacci numbers, run it, and tell me the output."
"""

from __future__ import annotations

import argparse
import sys
import time

import httpx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--mode", default="agent", choices=["auto", "agent", "team"])
    ap.add_argument("--key", default="")
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()
    h = {"x-api-key": a.key} if a.key else {}
    with httpx.Client(base_url=a.url, headers=h, timeout=60) as c:
        sid = c.post("/api/sessions", json={"config": {"mode": a.mode, "mcp": [], "memory": True}}).json()["id"]
        c.post(f"/api/sessions/{sid}/messages", json={"content": a.prompt}).raise_for_status()
        t0, seen = time.time(), 0
        while time.time() - t0 < a.timeout:
            time.sleep(3)
            msgs = c.get(f"/api/sessions/{sid}/messages").json()
            for m in msgs[seen:]:
                if m["role"] == "assistant":
                    calls = ", ".join(f"{t['name']}({', '.join(f'{k}={str(v)[:60]!r}' for k, v in t['args'].items())})" for t in m["tool_calls"])
                    print(f"[{m['agent']}] ASSISTANT: {m['content'][:300]!r} {('-> ' + calls) if calls else ''}", flush=True)
                elif m["role"] == "tool":
                    print(f"   TOOL {m['tool_name']}: {m['content'][:200]!r}", flush=True)
                elif m["role"] == "card":
                    print(f"   CARD {m['meta']['card']['kind']}: {m['meta']['card']['title']}", flush=True)
                elif m["role"] == "notice":
                    print(f"   NOTICE: {m['content']}", flush=True)
            seen = len(msgs)
            if not c.get(f"/api/sessions/{sid}/state").json()["running"]:
                break
        st = c.get(f"/api/sessions/{sid}").json()
        print("usage:", st["usage"], "| todos:", st["todos"])
        files = c.get(f"/api/sessions/{sid}/files").json()["entries"]
        print("files:", [f["name"] for f in files])
        print(f"elapsed {time.time() - t0:.0f}s  session={sid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
