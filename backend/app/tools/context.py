"""Per-run context handed to every tool (sandbox, stores, event bus, approvals)."""

from __future__ import annotations

import contextvars
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..config import Settings
from ..sandbox.base import Sandbox

# id of the tool call currently executing (lets tools stream output against the right card)
current_call_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_call_id", default=None)
current_agent: contextvars.ContextVar[str] = contextvars.ContextVar("current_agent", default="main")


class SessionStore(Protocol):
    """Durable per-session state used by tools (backed by PostgreSQL in production)."""

    async def get_todos(self) -> list[dict[str, Any]]: ...
    async def set_todos(self, todos: list[dict[str, Any]]) -> None: ...
    async def get_core_memory(self) -> dict[str, str]: ...
    async def set_core_block(self, block: str, text: str) -> None: ...


class MemoryBackend(Protocol):
    async def save(self, *, session_id: str | None, kind: str, content: str, tags: list[str] | None = None,
                   importance: int = 3, source: str = "agent") -> dict[str, Any]: ...
    async def search(self, query: str, *, session_id: str | None = None, kind: str | None = None,
                     limit: int = 5) -> list[dict[str, Any]]: ...
    async def forget(self, memory_id: str) -> bool: ...


class EventSink(Protocol):
    async def emit(self, type: str, **data: Any) -> None: ...


class ApprovalGate(Protocol):
    async def request(self, *, tool: str, args: dict[str, Any], call_id: str, agent: str) -> tuple[bool, str]:
        """Return (approved, reason)."""


@dataclass
class ToolContext:
    session_id: str
    sandbox: Sandbox
    settings: Settings
    store: SessionStore
    memory: MemoryBackend | None = None
    events: EventSink | None = None
    approvals: ApprovalGate | None = None
    approval_mode: str = "auto"  # auto | ask
    agent: str = "main"
    # set by the graph builder
    delegate: Callable[[str, str], Awaitable[str]] | None = None
    handoffs: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    async def emit(self, type: str, **data: Any) -> None:
        if self.events is not None:
            await self.events.emit(type, agent=current_agent.get(), **data)
