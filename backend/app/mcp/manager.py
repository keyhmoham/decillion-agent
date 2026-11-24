"""Per-session MCP connections.

Each MCP server runs inside the session's sandbox and is kept alive by a dedicated asyncio task
(the ``mcp`` stdio client uses anyio task groups, which must be entered and exited by the same
task).  Failures are isolated: one broken server never blocks a run.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from ..config import Settings
from ..sandbox.base import Sandbox
from .catalog import PRESETS

log = logging.getLogger(__name__)


@dataclass
class McpHandle:
    key: str
    name: str
    spec: dict[str, Any]
    tools: list[BaseTool] = field(default_factory=list)
    error: str | None = None
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    stop: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None


def spec_key(spec: dict[str, Any]) -> str:
    return repr(sorted((k, repr(v)) for k, v in spec.items()))


class McpManager:
    def __init__(self, settings: Settings, backend_env: dict[str, str] | None = None) -> None:
        self.s = settings
        self.env = backend_env if backend_env is not None else dict(os.environ)
        self._handles: dict[str, dict[str, McpHandle]] = {}  # session -> name -> handle
        self._locks: dict[str, asyncio.Lock] = {}

    # ------------------------------------------------------------------ spec building
    def _spec_for_preset(self, sandbox: Sandbox, preset_id: str) -> dict[str, Any] | None:
        p = PRESETS.get(preset_id)
        if p is None:
            return None
        if any(not self.env.get(r) for r in p.requires):
            return None
        env = {k: self.env[v] for k, v in p.env_from_settings.items() if self.env.get(v)}
        cmd, args, penv = sandbox.mcp_stdio(list(p.argv), env)
        spec: dict[str, Any] = {"transport": "stdio", "command": cmd, "args": args}
        if penv:
            spec["env"] = penv
        return spec

    def _spec_for_custom(self, sandbox: Sandbox, item: dict[str, Any]) -> dict[str, Any] | None:
        if item.get("url"):
            transport = item.get("transport", "streamable_http")
            spec: dict[str, Any] = {"transport": transport, "url": item["url"]}
            if item.get("headers"):
                spec["headers"] = item["headers"]
            return spec
        argv = item.get("command")
        if isinstance(argv, str):
            import shlex

            argv = shlex.split(argv)
        if not argv:
            return None
        cmd, args, penv = sandbox.mcp_stdio(list(argv), item.get("env") or {})
        spec = {"transport": "stdio", "command": cmd, "args": args}
        if penv:
            spec["env"] = penv
        return spec

    # ------------------------------------------------------------------ lifecycle
    async def _run_handle(self, h: McpHandle) -> None:
        try:
            client = MultiServerMCPClient({h.name: h.spec})
            async with client.session(h.name) as sess:
                h.tools = await load_mcp_tools(sess, server_name=h.name)
                h.ready.set()
                await h.stop.wait()
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001  (ExceptionGroup from anyio too)
            h.error = _brief(exc)
            log.warning("MCP server %s failed: %s", h.name, h.error)
        finally:
            h.ready.set()

    async def ensure(
        self, session_id: str, sandbox: Sandbox, selected: list[str], custom: list[dict[str, Any]] | None = None,
    ) -> tuple[list[BaseTool], dict[str, str]]:
        """Return (tools, errors-by-server). Starts missing servers, stops deselected ones."""
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            wanted: dict[str, dict[str, Any]] = {}
            errors: dict[str, str] = {}
            for pid in selected:
                spec = self._spec_for_preset(sandbox, pid)
                if spec is None:
                    errors[pid] = "unknown preset or missing credentials"
                else:
                    wanted[pid] = spec
            for item in custom or []:
                name = str(item.get("name") or "custom")[:32]
                spec = self._spec_for_custom(sandbox, item)
                if spec is None:
                    errors[name] = "invalid custom server definition"
                else:
                    wanted[name] = spec

            current = self._handles.setdefault(session_id, {})
            for name in list(current):
                if name not in wanted or current[name].key != spec_key(wanted[name]) or current[name].error:
                    await self._stop_handle(current.pop(name))
            new: list[McpHandle] = []
            for name, spec in wanted.items():
                if name not in current:
                    h = McpHandle(key=spec_key(spec), name=name, spec=spec)
                    h.task = asyncio.create_task(self._run_handle(h), name=f"mcp-{session_id[:6]}-{name}")
                    current[name] = h
                    new.append(h)
            for h in new:
                try:
                    await asyncio.wait_for(h.ready.wait(), timeout=self.s.mcp_startup_timeout)
                except asyncio.TimeoutError:
                    h.error = f"startup timed out after {self.s.mcp_startup_timeout}s"
            tools: list[BaseTool] = []
            for name, h in current.items():
                if h.error:
                    errors[name] = h.error
                else:
                    tools.extend(h.tools)
            return tools, errors

    async def _stop_handle(self, h: McpHandle) -> None:
        h.stop.set()
        if h.task:
            try:
                await asyncio.wait_for(h.task, timeout=5)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):  # noqa: BLE001
                h.task.cancel()

    async def close(self, session_id: str) -> None:
        for h in list(self._handles.pop(session_id, {}).values()):
            await self._stop_handle(h)

    async def close_all(self) -> None:
        for sid in list(self._handles):
            await self.close(sid)

    def status(self, session_id: str) -> list[dict[str, Any]]:
        return [
            {"name": h.name, "tools": [t.name for t in h.tools], "error": h.error}
            for h in self._handles.get(session_id, {}).values()
        ]


def _brief(exc: BaseException) -> str:
    if isinstance(exc, BaseExceptionGroup):
        return "; ".join(_brief(e) for e in exc.exceptions)[:300]
    return f"{type(exc).__name__}: {exc}"[:300]
