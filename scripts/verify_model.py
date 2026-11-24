#!/usr/bin/env python3
"""Verify that the whole prompt/parse/tool-calling stack matches the live model server.

    # against the compose stack (model server must be reachable from where you run this)
    LLM_BACKEND=ollama LLM_BASE_URL=http://localhost:11434 python scripts/verify_model.py
    # against llama.cpp / llama-cpp-python:
    LLM_BACKEND=openai LLM_BASE_URL=http://localhost:8080 LLM_MODEL=a1 python scripts/verify_model.py

Exit code 0 only if every check passes. (Inside the running stack the same checks are available at
POST /api/model/selftest and under Context -> "Run self-test" in the UI.)
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import Settings  # noqa: E402
from app.llm.backends import make_backend  # noqa: E402
from app.llm.chat_model import build_chat_model  # noqa: E402
from app.llm.selftest import run_selftest  # noqa: E402


async def main() -> int:
    s = Settings()
    if s.llm_base_url.startswith("http://ollama"):
        s = Settings(llm_base_url="http://localhost:11434")
    backend = make_backend(s)
    health = await backend.health()
    print(f"server: {health}")
    if not health.get("ok"):
        return 2
    model = build_chat_model(s, backend)
    print(f"model={s.llm_model} backend={s.llm_backend} exact-tokenizer={model._counter.exact}\n")
    results = await run_selftest(model)
    for r in results:
        print(f"[{'PASS' if r.passed else 'FAIL'}] {r.name:<32} {r.seconds:>6.1f}s  {r.detail[:110]}")
    await backend.aclose()
    ok = all(r.passed for r in results)
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
