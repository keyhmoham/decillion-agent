"""The single-agent LangGraph: a ReAct loop hardened for a 4B model.

    START -> agent -> (tools -> agent)* -> finish_check -> END
                \\-> format_fix / empty_fix -> agent        (recoverable model mistakes)
                \\-> finalize -> END                         (step budget / runaway guard)

Hardening that matters for Agents-A1-4B:
* malformed or unknown tool calls are answered with an actionable error, never a crash;
* arguments are validated against the tool's JSON schema *before* execution;
* identical repeated calls are detected and the model is told to change approach;
* the model cannot "finish" while its own plan still has open steps (max 2 push-backs);
* context compaction keeps long sessions inside the window the model was tuned for;
* dangling tool calls (cancelled runs) are repaired so the template always renders.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langchain_core.messages import (
    AIMessage, AnyMessage, BaseMessage, RemoveMessage, SystemMessage, ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages

from ..config import Settings
from ..llm.chat_model import A1ChatModel
from ..llm.toolcall import compact_tool_schema, resolve_tool_name, validate_arguments
from ..tools.base import truncate_middle
from ..tools.context import ToolContext, current_agent, current_call_id
from .compaction import compact, microcompact
from .events import RunRecorder
from .roles import Role

log = logging.getLogger(__name__)

TOOL_FORMAT_HELP = (
    "Call a tool with exactly this format:\n<tool_call>\n<function=TOOL_NAME>\n<parameter=PARAM_NAME>\nVALUE\n</parameter>\n</function>\n</tool_call>"
)


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    steps: int
    nudges: int
    format_errors: int
    empties: int
    sigs: list[str]


@dataclass
class AgentDeps:
    settings: Settings
    model: A1ChatModel
    recorder: RunRecorder
    ctx: ToolContext
    tools: list[BaseTool]
    system_prompt: str
    agent_name: str = "main"
    role: Role | None = None
    task_key: str | None = None
    max_steps: int = 40
    enforce_todos: bool = True
    thinking: bool = True
    thinking_budget: int | None = None
    temperature: float | None = None
    parallel_readonly: bool = True
    tool_timeout: int = 600
    extra_system: dict[str, str] = field(default_factory=dict)

    @property
    def tool_map(self) -> dict[str, BaseTool]:
        return {t.name: t for t in self.tools}


def _sig(name: str, args: dict[str, Any]) -> str:
    return hashlib.sha1(f"{name}:{json.dumps(args, sort_keys=True, default=str)}".encode()).hexdigest()[:12]


def _flatten(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def _nudge(text: str, tag: str) -> ToolMessage:
    """A harness message. Rendered by the template inside a <tool_response> user turn, which keeps
    the model's multi-step reasoning replay intact (a real user message would reset it)."""
    return ToolMessage(content=f"[system] {text}", tool_call_id=f"sys-{tag}", name="system")


def repair_dangling(messages: list[BaseMessage]) -> list[ToolMessage]:
    """Tool calls without a result (cancelled run / crash) get a synthetic result."""
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    fixes = []
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls:
                if tc["id"] not in answered:
                    fixes.append(ToolMessage(content="Error: this tool call was interrupted before it finished (run cancelled).",
                                             tool_call_id=tc["id"], name=tc["name"]))
    return fixes


def args_hint(tool: BaseTool) -> str:
    params = compact_tool_schema(convert_to_openai_tool(tool))["function"]["parameters"]
    props = params.get("properties", {})
    req = set(params.get("required", []))
    return ", ".join(f"{k}{'' if k in req else '?'}: {v.get('type', 'any')}" for k, v in props.items())


async def execute_tool_call(deps: AgentDeps, tc: dict[str, Any], config: RunnableConfig) -> ToolMessage:
    """Validate, authorise, run and record one tool call. Never raises."""
    name, args, call_id = tc["name"], tc.get("args") or {}, tc["id"]
    tools = deps.tool_map
    rec = deps.recorder
    started = time.monotonic()

    async def finish(text: str, ok: bool) -> ToolMessage:
        text = truncate_middle(text, deps.settings.tool_output_limit, "output shortened")
        await rec.tool_end(call_id, name, args, text, ok, time.monotonic() - started, deps.agent_name, deps.task_key)
        return ToolMessage(content=text, tool_call_id=call_id, name=name)

    await rec.tool_start(call_id, name, args, deps.agent_name, deps.task_key)

    tool = tools.get(name)
    if tool is None:
        suggestion = resolve_tool_name(name, list(tools))
        hint = f" Did you mean '{suggestion}'?" if suggestion else ""
        return await finish(f"Error: unknown tool '{name}'.{hint} Available tools: {', '.join(tools)}.", False)

    schema = convert_to_openai_tool(tool)["function"]["parameters"]
    problems = validate_arguments(args, schema)
    if problems:
        return await finish(
            f"Error: invalid arguments for {name}: " + "; ".join(problems) + f". Expected: {name}({args_hint(tool)}).", False
        )

    meta = tool.metadata or {}
    if deps.ctx.approval_mode == "ask" and meta.get("mutating") and deps.ctx.approvals is not None:
        approved, reason = await deps.ctx.approvals.request(tool=name, args=args, call_id=call_id, agent=deps.agent_name)
        if not approved:
            return await finish(f"The user denied this action{': ' + reason if reason else ''}. Do not retry it; choose another approach or ask the user.", False)

    token = current_call_id.set(call_id)
    agent_token = current_agent.set(deps.agent_name)
    try:
        result = await asyncio.wait_for(
            tool.ainvoke({"name": name, "args": args, "id": call_id, "type": "tool_call"}, config), timeout=deps.tool_timeout
        )
        content = result.content if isinstance(result, ToolMessage) else result
        text = _flatten(content)
        ok = not text.startswith("Error:")
        return await finish(text, ok)
    except asyncio.TimeoutError:
        return await finish(f"Error: tool '{name}' timed out after {deps.tool_timeout}s.", False)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("tool %s crashed", name)
        return await finish(f"Error: {type(exc).__name__}: {exc}", False)
    finally:
        current_call_id.reset(token)
        current_agent.reset(agent_token)


def build_agent_graph(deps: AgentDeps, checkpointer: Any = None):
    s = deps.settings
    model = deps.model
    tools = deps.tools
    bound = model.bind_tools(tools) if tools else model
    agent_cfg = {"agent": deps.agent_name, "task_key": deps.task_key}

    def llm_config(config: RunnableConfig, **meta: Any) -> RunnableConfig:
        return {**config, "metadata": {**(config.get("metadata") or {}), **agent_cfg, **meta}}

    def system_msg() -> SystemMessage:
        return SystemMessage(content=deps.system_prompt)

    def count(msgs: list[BaseMessage]) -> int:
        return model.count_prompt_tokens([system_msg(), *msgs], tools)

    llm_overrides: dict[str, Any] = {"enable_thinking": deps.thinking}
    if deps.thinking_budget is not None:
        llm_overrides["thinking_budget"] = deps.thinking_budget
    if deps.temperature is not None:
        llm_overrides["temperature"] = deps.temperature
    runnable = bound.bind(**llm_overrides)

    # ------------------------------------------------------------------ nodes
    async def agent_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        msgs: list[BaseMessage] = list(state.get("messages", []))
        updates: list[BaseMessage] = []

        fixes = repair_dangling(msgs)
        if fixes:
            updates += fixes
            msgs += fixes

        tokens = count(msgs)
        budget = s.context_budget
        if tokens > budget * 0.7:
            trimmed = microcompact(msgs)
            if trimmed:
                updates += trimmed  # same ids -> replaced in place by add_messages
                by_id = {t.id: t for t in trimmed}
                msgs = [by_id.get(m.id, m) for m in msgs]
                tokens = count(msgs)
        if tokens > budget:
            result = await compact(model, msgs, count_tokens=count, config=llm_config(config))
            if result is not None:
                await deps.recorder.emit("compaction", agent=deps.agent_name, tokens_before=result.tokens_before,
                                         tokens_after=result.tokens_after, summary=result.summary)
                updates = [RemoveMessage(id=REMOVE_ALL_MESSAGES), *result.messages]
                msgs = list(result.messages)
                tokens = result.tokens_after
        await deps.recorder.emit("context", agent=deps.agent_name, tokens=tokens, budget=budget)

        ai: AIMessage = await runnable.ainvoke([system_msg(), *msgs], llm_config(config))
        errors = ai.additional_kwargs.get("tool_call_errors") or []
        if errors and not ai.tool_calls:
            raw = "\n".join(e["raw"][:600] for e in errors)
            ai = AIMessage(content=(ai.content + "\n" if ai.content else "") + (f"<tool_call>\n{raw}\n</tool_call>" if raw else ""),
                           additional_kwargs=ai.additional_kwargs, response_metadata=ai.response_metadata,
                           usage_metadata=ai.usage_metadata)
        return {"messages": [*updates, ai], "steps": state.get("steps", 0) + 1}

    def route_after_agent(state: AgentState) -> str:
        ai = state["messages"][-1]
        assert isinstance(ai, AIMessage)
        if state.get("steps", 0) >= deps.max_steps:
            return "finalize"
        if ai.tool_calls:
            return "tools"
        if ai.additional_kwargs.get("tool_call_errors"):
            return "format_fix" if state.get("format_errors", 0) < 3 else "finalize"
        if not _flatten(ai.content).strip():
            return "empty_fix" if state.get("empties", 0) < 2 else "finalize"
        return "finish_check"

    async def tools_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        ai = state["messages"][-1]
        assert isinstance(ai, AIMessage)
        calls = list(ai.tool_calls)
        sigs = list(state.get("sigs", []))

        readonly = all(not (deps.tool_map.get(c["name"]).metadata or {}).get("mutating") for c in calls if c["name"] in deps.tool_map)
        if deps.parallel_readonly and len(calls) > 1 and readonly:
            results = await asyncio.gather(*(execute_tool_call(deps, c, config) for c in calls))
        else:
            results = []
            for c in calls:
                results.append(await execute_tool_call(deps, c, config))

        out: list[ToolMessage] = []
        for c, r in zip(calls, results):
            sig = _sig(c["name"], c.get("args") or {})
            repeats = sigs[-8:].count(sig)
            sigs.append(sig)
            if repeats >= 2:
                r = ToolMessage(
                    content=r.content + f"\n[system] You have made this exact call {repeats + 1} times. Repeating it will not help. "
                                         "Change your approach, or if you are blocked, explain the problem in your final answer.",
                    tool_call_id=r.tool_call_id, name=r.name,
                )
            out.append(r)
        return {"messages": out, "sigs": sigs[-24:]}

    def route_after_tools(state: AgentState) -> str:
        recent = state.get("sigs", [])[-8:]
        if recent and max(recent.count(x) for x in set(recent)) >= 5:
            return "finalize"  # hard stop on a stuck loop
        if state.get("steps", 0) >= deps.max_steps:
            return "finalize"
        return "agent"

    async def format_fix(state: AgentState) -> dict[str, Any]:
        ai = state["messages"][-1]
        errs = ai.additional_kwargs.get("tool_call_errors", [])
        why = errs[0]["error"] if errs else "unreadable tool call"
        msg = _nudge(f"Your tool call could not be parsed: {why}\n{TOOL_FORMAT_HELP}\nTry again, or answer without a tool if none is needed.", f"fmt{state.get('format_errors', 0)}")
        await deps.recorder.emit("notice", level="warn", agent=deps.agent_name, text=f"Malformed tool call, asking the model to retry ({why})")
        return {"messages": [msg], "format_errors": state.get("format_errors", 0) + 1}

    async def empty_fix(state: AgentState) -> dict[str, Any]:
        msg = _nudge("Your last reply was empty. Either call a tool to make progress, or write your final answer.", f"empty{state.get('empties', 0)}")
        return {"messages": [msg], "empties": state.get("empties", 0) + 1}

    async def finish_check(state: AgentState) -> dict[str, Any]:
        if not deps.enforce_todos or state.get("nudges", 0) >= 2:
            return {}
        todos = await deps.ctx.store.get_todos()
        open_items = [t for t in todos if t["status"] != "completed"]
        if not open_items:
            return {}
        listing = "; ".join(t["content"] for t in open_items[:5])
        msg = _nudge(
            f"You wrote a plan but these steps are not marked completed: {listing}. Finish them (then call update_todo), "
            "or if they are no longer needed mark them completed and explain briefly.", f"todo{state.get('nudges', 0)}",
        )
        return {"messages": [msg], "nudges": state.get("nudges", 0) + 1}

    def route_after_finish_check(state: AgentState) -> str:
        return "agent" if isinstance(state["messages"][-1], ToolMessage) else END

    async def finalize(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        await deps.recorder.emit("notice", level="warn", agent=deps.agent_name, text="Step budget or loop guard reached - summarising progress.")
        msgs = list(state["messages"])
        if msgs and isinstance(msgs[-1], AIMessage) and msgs[-1].tool_calls:
            msgs += [ToolMessage(content="Error: not executed (step limit).", tool_call_id=c["id"], name=c["name"]) for c in msgs[-1].tool_calls]
        msgs.append(_nudge("You are out of steps. Do not call tools. Summarise what you accomplished, what is unfinished and what the user should do next.", "final"))
        ai = await model.bind(enable_thinking=False, max_tokens=800).ainvoke([system_msg(), *msgs], llm_config(config))
        extra = [m for m in msgs[len(state["messages"]):]]
        return {"messages": [*extra, ai]}

    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_node("format_fix", format_fix)
    g.add_node("empty_fix", empty_fix)
    g.add_node("finish_check", finish_check)
    g.add_node("finalize", finalize)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route_after_agent, ["tools", "format_fix", "empty_fix", "finish_check", "finalize"])
    g.add_conditional_edges("tools", route_after_tools, ["agent", "finalize"])
    g.add_edge("format_fix", "agent")
    g.add_edge("empty_fix", "agent")
    g.add_conditional_edges("finish_check", route_after_finish_check, ["agent", END])
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


def final_text(messages: list[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and _flatten(m.content).strip():
            return _flatten(m.content).strip()
    return ""
