from __future__ import annotations

import json
from typing import Any

from app.agents.events import EventBus, RunRecorder
from app.config import Settings
from app.db.repo import InMemorySessionStore
from app.sandbox.local_sandbox import LocalSandbox
from app.tools.context import ToolContext


def call_text(name: str, **args: Any) -> str:
    parts = ["<tool_call>", f"<function={name}>"]
    for k, v in args.items():
        sv = v if isinstance(v, str) else json.dumps(v)
        parts += [f"<parameter={k}>", sv, "</parameter>"]
    parts += ["</function>", "</tool_call>"]
    return "\n".join(parts)


def turn(text: str = "", think: str = "ok", **call: Any) -> str:
    out = f"{think}\n</think>\n\n{text}"
    for name, args in call.items():
        out += "\n" + call_text(name, **args)
    return out


def make_ctx(tmp_path, **settings_kw) -> tuple[ToolContext, EventBus, RunRecorder, Settings]:
    s = Settings(sandbox_backend="local", sandbox_local_root=str(tmp_path / "sbx"), **settings_kw)
    sb = LocalSandbox("test-session-0001", s)
    bus = EventBus("test-session-0001")
    rec = RunRecorder(bus, None, "test-session-0001", "00000000-0000-0000-0000-000000000001")
    ctx = ToolContext(session_id="test-session-0001", sandbox=sb, settings=s, store=InMemorySessionStore(), events=rec)
    return ctx, bus, rec, s
