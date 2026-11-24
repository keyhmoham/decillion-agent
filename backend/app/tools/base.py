"""Helpers for building tools that suit a 4B model: short names, strict schemas, readable errors."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from ..llm.toolcall import compact_tool_schema  # noqa: F401  (re-export)


class ToolError(Exception):
    """Raised by tools to return a clean, actionable error message to the model."""


def truncate_middle(text: str, limit: int, hint: str = "") -> str:
    if len(text) <= limit:
        return text
    head = int(limit * 0.6)
    tail = limit - head
    omitted = len(text) - head - tail
    note = f"\n[... {omitted} characters omitted{'; ' + hint if hint else ''} ...]\n"
    return text[:head] + note + text[-tail:]


def make_tool(
    name: str,
    description: str,
    args_schema: type[BaseModel],
    coroutine: Callable[..., Awaitable[str]],
    *,
    mutating: bool = False,
    category: str = "general",
) -> StructuredTool:
    async def _runner(**kwargs: Any) -> str:
        try:
            return await coroutine(**kwargs)
        except ToolError as exc:
            return f"Error: {exc}"

    return StructuredTool(
        name=name,
        description=description,
        args_schema=args_schema,
        coroutine=_runner,
        metadata={"mutating": mutating, "category": category},
        handle_tool_error=True,
        handle_validation_error=True,
    )
