"""Event bus + recorder: the single place where agent activity becomes UI events and DB rows."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from ..db.repo import Repo

PersistHook = Callable[[dict[str, Any]], Awaitable[None]]


class EventBus:
    """Per-session pub/sub with a replay buffer, so a reloaded browser tab catches up."""

    def __init__(self, session_id: str, buffer_size: int = 4000) -> None:
        self.session_id = session_id
        self.seq = 0
        self.buffer: deque[dict[str, Any]] = deque(maxlen=buffer_size)
        self.subs: set[asyncio.Queue] = set()
        self.persist_hook: PersistHook | None = None

    async def emit(self, type: str, **data: Any) -> dict[str, Any]:
        self.seq += 1
        evt = {"seq": self.seq, "type": type, "ts": time.time(), "session_id": self.session_id, **data}
        self.buffer.append(evt)
        for q in list(self.subs):
            try:
                q.put_nowait(evt)
            except asyncio.QueueFull:  # slow consumer: drop it, it will resync via REST
                self.subs.discard(q)
        if self.persist_hook is not None:
            try:
                await self.persist_hook(evt)
            except Exception:  # noqa: BLE001
                import logging

                logging.getLogger(__name__).exception("event persistence failed")
        return evt

    def subscribe(self, since: int = 0) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=10000)
        for evt in self.buffer:
            if evt["seq"] > since:
                q.put_nowait(evt)
        self.subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self.subs.discard(q)


class RunRecorder:
    """Records a run: persists messages and forwards everything to the bus."""

    def __init__(self, bus: EventBus, repo: Repo | None, session_id: str, run_id: str) -> None:
        self.bus, self.repo, self.session_id, self.run_id = bus, repo, session_id, run_id
        self.usage = {"input_tokens": 0, "output_tokens": 0, "llm_calls": 0, "tool_calls": 0}

    async def emit(self, type: str, **data: Any) -> None:
        await self.bus.emit(type, run_id=self.run_id, **data)

    async def persist(self, role: str, **fields: Any) -> dict[str, Any] | None:
        if self.repo is None:
            return None
        return await self.repo.add_message(self.session_id, role, run_id=UUID(self.run_id), **fields)

    # tool lifecycle ---------------------------------------------------------------------
    async def tool_start(self, call_id: str, name: str, args: dict[str, Any], agent: str, task_key: str | None) -> None:
        self.usage["tool_calls"] += 1
        await self.emit("tool_start", call_id=call_id, name=name, args=args, agent=agent, task_key=task_key)

    async def tool_end(self, call_id: str, name: str, args: dict[str, Any], result: str, ok: bool, duration: float,
                       agent: str, task_key: str | None) -> None:
        await self.emit("tool_end", call_id=call_id, name=name, result=result, ok=ok, duration=round(duration, 2), agent=agent, task_key=task_key)
        await self.persist("tool", agent=agent, content=result, tool_call_id=call_id, tool_name=name,
                           meta={"args": args, "ok": ok, "duration": round(duration, 2), "task_key": task_key})


class RunCallbacks(AsyncCallbackHandler):
    """LangChain callbacks -> streaming message events + persisted assistant messages."""

    run_inline = True

    def __init__(self, rec: RunRecorder) -> None:
        self.rec = rec
        self._calls: dict[UUID, dict[str, Any]] = {}

    async def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        meta = metadata or {}
        internal = "internal" in (tags or [])
        info = {
            "id": uuid.uuid4().hex[:12], "agent": meta.get("agent", "main"), "task_key": meta.get("task_key"),
            "internal": internal, "purpose": meta.get("purpose"),
        }
        self._calls[run_id] = info
        if not internal:
            await self.rec.emit("message_start", message_id=info["id"], agent=info["agent"], task_key=info["task_key"])
        else:
            await self.rec.emit("status", text=meta.get("purpose") or "working", agent=info["agent"])

    async def on_llm_new_token(self, token, *, chunk=None, run_id, parent_run_id=None, **kwargs):
        info = self._calls.get(run_id)
        if not info or info["internal"] or chunk is None:
            return
        msg = chunk.message
        reasoning = msg.additional_kwargs.get("reasoning_content") if hasattr(msg, "additional_kwargs") else None
        if reasoning:
            await self.rec.emit("reasoning_delta", message_id=info["id"], text=reasoning)
        text = chunk.text
        if text:
            await self.rec.emit("content_delta", message_id=info["id"], text=text)

    async def on_llm_end(self, response: LLMResult, *, run_id, parent_run_id=None, **kwargs):
        info = self._calls.pop(run_id, None)
        if not info:
            return
        gen = response.generations[0][0] if response.generations and response.generations[0] else None
        msg = gen.message if isinstance(gen, ChatGeneration) else None
        if not isinstance(msg, AIMessage):
            return
        usage = msg.usage_metadata or {}
        self.rec.usage["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
        self.rec.usage["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
        self.rec.usage["llm_calls"] += 1
        await self.rec.emit("usage", usage=dict(self.rec.usage), last_prompt_tokens=int(usage.get("input_tokens", 0) or 0))
        if info["internal"]:
            return
        reasoning = msg.additional_kwargs.get("reasoning_content", "") or ""
        calls = [{"id": tc["id"], "name": tc["name"], "args": tc["args"]} for tc in msg.tool_calls]
        flags = {k: v for k, v in (msg.response_metadata or {}).items() if k in (
            "thinking_forced_closed", "thinking_degenerate", "nothink_fallback", "reasoning_tokens", "finish_reason")}
        await self.rec.emit(
            "message_end", message_id=info["id"], agent=info["agent"], task_key=info["task_key"],
            content=msg.content if isinstance(msg.content, str) else "", reasoning=reasoning, tool_calls=calls, flags=flags,
        )
        if msg.content or reasoning or calls:
            await self.rec.persist(
                "assistant", agent=info["agent"], content=msg.content if isinstance(msg.content, str) else "",
                reasoning=reasoning, tool_calls=calls, meta={"flags": flags, "task_key": info["task_key"], "usage": dict(usage)},
            )

    async def on_llm_error(self, error, *, run_id, **kwargs):
        info = self._calls.pop(run_id, None)
        if info and not info["internal"]:
            await self.rec.emit("message_error", message_id=info["id"], error=str(error))
