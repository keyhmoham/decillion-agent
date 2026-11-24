"""Assemble the tool list for an agent from named groups."""

from __future__ import annotations

from langchain_core.tools import BaseTool

from .card_tools import build_card_tools
from .context import ToolContext
from .memory_tools import build_memory_tools
from .plan_tools import build_plan_tools
from .sandbox_tools import build_sandbox_tools
from .team_tools import build_delegate_tool, build_handoff_tool
from .web_tools import build_web_tools

READONLY = {"read_file", "list_dir", "glob", "grep"}


def build_tools(
    ctx: ToolContext,
    groups: tuple[str, ...] | list[str],
    *,
    deny: tuple[str, ...] = (),
    extra: list[BaseTool] | None = None,
    max_tools: int = 28,
) -> list[BaseTool]:
    sandbox = {t.name: t for t in build_sandbox_tools(ctx)}
    tools: dict[str, BaseTool] = {}

    def add(items) -> None:
        for t in items:
            tools.setdefault(t.name, t)

    for g in groups:
        if g == "sandbox":
            add(sandbox.values())
        elif g == "readonly":
            add(t for n, t in sandbox.items() if n in READONLY)
        elif g == "bash_ro":
            add([sandbox["bash"]])
        elif g == "web" and ctx.settings.enable_web_tools:
            add(build_web_tools(ctx))
        elif g == "web_read" and ctx.settings.enable_web_tools:
            add(t for t in build_web_tools(ctx) if t.name == "web_fetch")
        elif g == "present":
            add(build_card_tools(ctx))
        elif g == "present_min":
            add(t for t in build_card_tools(ctx) if t.name in {"show_table", "show_chart", "show_stats"})
        elif g == "plan":
            add(build_plan_tools(ctx))
        elif g == "memory":
            add(build_memory_tools(ctx))
        elif g == "memory_read":
            add(t for t in build_memory_tools(ctx) if t.name == "memory_search")
        elif g == "team":
            add(build_delegate_tool(ctx))
        elif g == "handoff":
            add(build_handoff_tool(ctx))
    for t in extra or []:
        if t.name in tools:  # name clash with a native tool: namespace the MCP one
            t = t.model_copy(update={"name": f"mcp_{t.name}"})
        tools.setdefault(t.name, t)
    for name in deny:
        tools.pop(name, None)
    result = list(tools.values())
    if len(result) > max_tools:
        # Native tools first, MCP tools get trimmed: a 4B model degrades with very long tool lists.
        result = result[:max_tools]
    return result
