"""Worker agents: the same hardened ReAct graph, specialised by role."""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from ..db.repo import InMemorySessionStore
from ..tools.base import truncate_middle
from ..tools.context import current_agent
from ..tools.registry import build_tools
from .env import RunEnv
from .graph import AgentDeps, build_agent_graph, final_text
from .prompts import build_system_prompt
from .roles import get_role

log = logging.getLogger(__name__)


@dataclass
class WorkerResult:
    text: str
    ok: bool = True
    steps: int = 0
    handoffs: list[dict[str, Any]] = field(default_factory=list)


async def run_worker(
    env: RunEnv, role_name: str, task_prompt: str, *, agent_name: str | None = None, task_key: str | None = None,
    config: RunnableConfig | None = None,
) -> WorkerResult:
    role = get_role(role_name)
    s = env.settings
    name = agent_name or role.name
    # Workers get their own todo scratch space and no further delegation (depth 1).
    wctx = dataclasses.replace(env.ctx, agent=name, handoffs=[], delegate=None, store=InMemorySessionStore(), extra=dict(env.ctx.extra))
    groups = tuple(role.tool_groups) + ("handoff",)
    extra = env.mcp_tools if "sandbox" in role.tool_groups else []
    tools = build_tools(wctx, groups, deny=role.deny_tools, extra=extra)
    deps = AgentDeps(
        settings=s, model=env.model, recorder=env.recorder, ctx=wctx, tools=tools,
        system_prompt=build_system_prompt(
            agent=name, workdir=wctx.sandbox.workdir, role=role, core_memory=env.core_memory, memory_enabled=False,
            extra="You are one member of an agent team. Your final message is your report to the orchestrator: list files created or changed, "
                  "how you verified the work, and anything unresolved.",
        ),
        agent_name=name, role=role, task_key=task_key, max_steps=min(role.max_steps, s.worker_max_steps),
        enforce_todos=False, thinking=env.thinking and role.thinking,
        thinking_budget=role.thinking_budget, temperature=role.temperature,
    )
    graph = build_agent_graph(deps)
    token = current_agent.set(name)
    try:
        cfg: RunnableConfig = {**(config or {}), "recursion_limit": deps.max_steps * 3 + 12}
        out = await graph.ainvoke({"messages": [HumanMessage(content=task_prompt)]}, cfg)
    except Exception as exc:  # noqa: BLE001
        if exc.__class__.__name__ in {"CancelledError"}:
            raise
        log.exception("worker %s failed", name)
        return WorkerResult(f"Worker crashed: {type(exc).__name__}: {exc}", ok=False)
    finally:
        current_agent.reset(token)
    text = final_text(out["messages"]) or "(the agent produced no final report)"
    return WorkerResult(text, ok=True, steps=out.get("steps", 0), handoffs=list(wctx.handoffs))


def make_delegate(env: RunEnv, config_getter=lambda: None):
    """Implements the ``delegate_task`` tool for the main agent."""
    counter = {"n": 0}

    async def delegate(role: str, task: str) -> str:
        counter["n"] += 1
        name = f"{role}-{counter['n']}"
        await env.recorder.emit("delegation", agent="main", to=name, role=role, task=task[:400])
        res = await run_worker(env, role, task, agent_name=name, config=config_getter())
        await env.recorder.emit("delegation_done", agent="main", to=name, ok=res.ok)
        return truncate_middle(f"Report from {name}:\n{res.text}", env.settings.tool_output_limit)

    return delegate
