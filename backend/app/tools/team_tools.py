"""Tools for delegation (sub-agents) and swarm-style hand-offs."""

from __future__ import annotations

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from ..agents.roles import ROLE_NAMES
from .base import ToolError, make_tool
from .context import ToolContext


class DelegateArgs(BaseModel):
    role: str = Field(description=f"Specialist to delegate to: {', '.join(ROLE_NAMES)}.")
    task: str = Field(description="Complete, self-contained instructions. The specialist cannot see this conversation.")


class HandoffArgs(BaseModel):
    role: str = Field(description=f"Agent that should continue the work: {', '.join(ROLE_NAMES)}.")
    reason: str = Field(description="Why that agent is a better fit and what it must do next.")


def build_delegate_tool(ctx: ToolContext) -> list[BaseTool]:
    if ctx.delegate is None:
        return []

    async def delegate_task(role: str, task: str) -> str:
        if role not in ROLE_NAMES:
            raise ToolError(f"Unknown role '{role}'. Choose one of: {', '.join(ROLE_NAMES)}.")
        assert ctx.delegate is not None
        return await ctx.delegate(role, task)

    return [make_tool(
        "delegate_task",
        "Run a specialist sub-agent (researcher, coder, tester, reviewer ...) on a self-contained job and get its report back.",
        DelegateArgs, delegate_task, category="team",
    )]


def build_handoff_tool(ctx: ToolContext) -> list[BaseTool]:
    async def handoff(role: str, reason: str) -> str:
        if role not in ROLE_NAMES:
            raise ToolError(f"Unknown role '{role}'. Choose one of: {', '.join(ROLE_NAMES)}.")
        ctx.handoffs.append({"role": role, "reason": reason})
        await ctx.emit("handoff", to=role, reason=reason)
        return f"Hand-off to '{role}' recorded. Write a short summary of what you did so far as your final message."

    return [make_tool("handoff", "Pass the rest of your task to a better-suited teammate (swarm hand-off). Then finish with a summary.", HandoffArgs, handoff, category="team")]
