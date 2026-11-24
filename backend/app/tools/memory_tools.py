"""Agentic memory tools (MemGPT-style): core blocks that are always in context + searchable archive."""

from __future__ import annotations

from typing import Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .base import ToolError, make_tool
from .context import ToolContext

CORE_BLOCKS = ("user", "project", "notes")
CORE_BLOCK_LIMIT = 1200


class SearchArgs(BaseModel):
    query: str = Field(description="What to look for, in natural language.")
    kind: Literal["fact", "preference", "episode", "lesson", "project"] | None = Field(None, description="Optional filter.")
    limit: int = Field(5, description="Max results (1-10).")


class SaveArgs(BaseModel):
    content: str = Field(description="One self-contained fact, preference or lesson (a full sentence).")
    kind: Literal["fact", "preference", "episode", "lesson", "project"] = Field("fact", description="Memory type.")
    importance: int = Field(3, description="1 (trivia) to 5 (critical).")
    scope: Literal["session", "global"] = Field("session", description="global = visible in all future sessions.")
    tags: list[str] = Field(default_factory=list, description="Optional keywords.")


class ForgetArgs(BaseModel):
    memory_id: str = Field(description="Id from memory_search results.")


class CoreArgs(BaseModel):
    block: Literal["user", "project", "notes"] = Field(description="user = who the user is/prefers; project = goals, stack, conventions; notes = scratchpad.")
    content: str = Field(description=f"Full new text of the block (max {CORE_BLOCK_LIMIT} chars). It replaces the old text.")


def render_memories(items: list[dict]) -> str:
    if not items:
        return "No matching memories."
    return "\n".join(
        f"- [{m['id'][:8]}] ({m['kind']}, importance {m['importance']}{', global' if m['session_id'] is None else ''}) {m['content']}"
        for m in items
    )


def build_memory_tools(ctx: ToolContext) -> list[BaseTool]:
    mem = ctx.memory
    if mem is None:
        return []

    async def memory_search(query: str, kind: str | None = None, limit: int = 5) -> str:
        items = await mem.search(query, session_id=ctx.session_id, kind=kind, limit=max(1, min(int(limit), 10)))
        await ctx.emit("memory", action="search", query=query, count=len(items))
        return render_memories(items)

    async def memory_save(content: str, kind: str = "fact", importance: int = 3, scope: str = "session", tags: list[str] | None = None) -> str:
        saved = await mem.save(
            session_id=None if scope == "global" else ctx.session_id, kind=kind, content=content,
            tags=tags or [], importance=importance, source="agent",
        )
        await ctx.emit("memory", action="save", memory=saved)
        return f"Saved memory {saved['id'][:8]}{' (merged with a similar one)' if saved.get('deduplicated') else ''}."

    async def memory_forget(memory_id: str) -> str:
        full = memory_id
        if len(memory_id) < 36 and hasattr(mem, "resolve_prefix"):
            full = await mem.resolve_prefix(memory_id) or memory_id  # type: ignore[attr-defined]
        if not await mem.forget(full):
            raise ToolError(f"No memory with id {memory_id}.")
        await ctx.emit("memory", action="forget", id=full)
        return "Memory deleted."

    async def core_memory_update(block: str, content: str) -> str:
        if block not in CORE_BLOCKS:
            raise ToolError(f"block must be one of {', '.join(CORE_BLOCKS)}")
        content = content.strip()
        if len(content) > CORE_BLOCK_LIMIT:
            raise ToolError(f"Too long ({len(content)} chars). Condense it to {CORE_BLOCK_LIMIT} characters or fewer.")
        await ctx.store.set_core_block(block, content)
        await ctx.emit("core_memory", block=block, content=content)
        return f"Core memory block '{block}' updated ({len(content)} chars)."

    return [
        make_tool("memory_search", "Search your long-term memory for facts, preferences and lessons from this and earlier sessions.", SearchArgs, memory_search, category="memory"),
        make_tool("memory_save", "Store something worth remembering later (user preferences, project facts, lessons learned).", SaveArgs, memory_save, category="memory"),
        make_tool("memory_forget", "Delete an outdated or wrong memory.", ForgetArgs, memory_forget, category="memory"),
        make_tool(
            "core_memory_update",
            "Rewrite one of the always-visible core memory blocks (user / project / notes). Keep them short.",
            CoreArgs, core_memory_update, category="memory",
        ),
    ]
