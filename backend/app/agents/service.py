"""AgentService: runs one chat turn end-to-end (memory recall, MCP, mode selection, graph, persistence)."""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..config import Settings
from ..db.repo import DbSessionStore, Repo
from ..llm.backends import BackendError, LLMBackend
from ..llm.chat_model import A1ChatModel, build_chat_model
from ..mcp.manager import McpManager
from ..memory.store import MemoryStore
from ..sandbox.manager import SandboxManager
from ..tools.base import truncate_middle
from ..tools.context import ToolContext, current_agent
from ..tools.registry import build_tools
from .compaction import transcript
from .env import RunEnv
from .events import EventBus, RunCallbacks, RunRecorder
from .graph import AgentDeps, build_agent_graph, final_text
from .prompts import REFLECT_PROMPT, TRIAGE_PROMPT, build_system_prompt
from .roles import GENERALIST
from .team import build_team_graph
from .workers import make_delegate

log = logging.getLogger(__name__)

TEAM_HINT = re.compile(r"\b(team|swarm|multi[- ]agent|agents? in parallel|orchestrat)", re.I)


class Triage(BaseModel):
    """Classification of the request."""

    complexity: str = Field(description="simple | moderate | complex")
    parallelizable: bool = Field(description="true if the work splits into independent parts")


class MemItem(BaseModel):
    content: str = Field(description="One short, self-contained memory.")
    kind: str = Field("fact", description="fact | preference | lesson | project")
    importance: int = Field(3, description="1-5")


class MemoryExtraction(BaseModel):
    """Memories worth keeping for future sessions."""

    memories: list[MemItem] = Field(default_factory=list, description="0-4 memories")


class ApprovalManager:
    """Human-in-the-loop gate for mutating tools (``approval = ask`` sessions)."""

    def __init__(self, bus: EventBus, run_id: str, timeout: float = 1800.0) -> None:
        self.bus, self.run_id, self.timeout = bus, run_id, timeout
        self.pending: dict[str, asyncio.Future] = {}

    async def request(self, *, tool: str, args: dict[str, Any], call_id: str, agent: str) -> tuple[bool, str]:
        aid = uuid.uuid4().hex[:10]
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self.pending[aid] = fut
        await self.bus.emit("approval_request", run_id=self.run_id, approval_id=aid, tool=tool, args=args, call_id=call_id, agent=agent)
        try:
            approved, note = await asyncio.wait_for(fut, self.timeout)
        except asyncio.TimeoutError:
            approved, note = False, "approval timed out"
        finally:
            self.pending.pop(aid, None)
        await self.bus.emit("approval_resolved", run_id=self.run_id, approval_id=aid, approved=approved)
        return approved, note

    def resolve(self, aid: str, approved: bool, note: str = "") -> bool:
        fut = self.pending.get(aid)
        if fut and not fut.done():
            fut.set_result((approved, note))
            return True
        return False

    def deny_all(self, note: str = "run cancelled") -> None:
        for fut in self.pending.values():
            if not fut.done():
                fut.set_result((False, note))


@dataclass
class RunHandle:
    run_id: str
    task: asyncio.Task
    approvals: ApprovalManager | None = None
    mode: str = "agent"


@dataclass
class AgentService:
    settings: Settings
    repo: Repo
    memory: MemoryStore
    sandboxes: SandboxManager
    mcp: McpManager
    backend: LLMBackend
    checkpointer: Any = None
    buses: dict[str, EventBus] = field(default_factory=dict)
    runs: dict[str, RunHandle] = field(default_factory=dict)
    mcp_warned: dict[str, set] = field(default_factory=dict)
    search: Any = None

    # ------------------------------------------------------------------ helpers
    def bus(self, session_id: str) -> EventBus:
        b = self.buses.get(session_id)
        if b is None:
            b = self.buses[session_id] = EventBus(session_id)
            b.persist_hook = self._make_persist_hook(session_id)
        return b

    def _make_persist_hook(self, session_id: str):
        async def hook(evt: dict[str, Any]) -> None:
            t = evt["type"]
            run_id = evt.get("run_id")
            rid = uuid.UUID(run_id) if run_id else None
            if t == "card":
                await self.repo.add_message(session_id, "card", run_id=rid, agent=evt.get("agent", "main"), meta={"card": evt["card"]})
            elif t == "notice":
                await self.repo.add_message(session_id, "notice", run_id=rid, agent=evt.get("agent", "main"), content=evt.get("text", ""), meta={"level": evt.get("level", "info")})
            elif t == "compaction":
                await self.repo.set_fields(session_id, summary=evt.get("summary", ""))
                await self.repo.add_message(session_id, "notice", run_id=rid, content=f"Context compacted ({evt['tokens_before']} → {evt['tokens_after']} tokens).", meta={"level": "info", "compaction": True})
        return hook

    def is_running(self, session_id: str) -> bool:
        h = self.runs.get(session_id)
        return bool(h and not h.task.done())

    def current_run_id(self, session_id: str) -> str | None:
        h = self.runs.get(session_id)
        return h.run_id if h and not h.task.done() else None

    def make_model(self, **overrides: Any) -> A1ChatModel:
        return build_chat_model(self.settings, self.backend, **overrides)

    # ------------------------------------------------------------------ public API
    async def start_run(self, session_id: str, text: str, overrides: dict[str, Any] | None = None) -> str:
        if self.is_running(session_id):
            raise RuntimeError("A run is already active in this session.")
        session = await self.repo.get_session(session_id)
        if session is None:
            raise KeyError(session_id)
        cfg = {**session["config"], **(overrides or {})}
        if overrides:
            await self.repo.update_session(session_id, config=overrides)
        mode = cfg.get("mode", self.settings.default_mode)
        run_id = await self.repo.create_run(session_id, mode, text)
        user_msg = await self.repo.add_message(session_id, "user", run_id=uuid.UUID(run_id), content=text)
        bus = self.bus(session_id)
        await bus.emit("user_message", run_id=run_id, message=user_msg)
        task = asyncio.create_task(self._execute(session_id, run_id, text, cfg, session), name=f"run-{run_id[:8]}")
        self.runs[session_id] = RunHandle(run_id, task, mode=mode)
        return run_id

    async def cancel(self, session_id: str) -> bool:
        h = self.runs.get(session_id)
        if not h or h.task.done():
            return False
        if h.approvals:
            h.approvals.deny_all()
        h.task.cancel()
        return True

    def resolve_approval(self, session_id: str, approval_id: str, approve: bool, note: str = "") -> bool:
        h = self.runs.get(session_id)
        return bool(h and h.approvals and h.approvals.resolve(approval_id, approve, note))

    async def delete_session(self, session_id: str) -> None:
        await self.cancel(session_id)
        h = self.runs.pop(session_id, None)
        if h:
            try:
                await asyncio.wait_for(h.task, 10)
            except Exception:  # noqa: BLE001
                pass
        await self.mcp.close(session_id)
        await self.sandboxes.destroy(session_id)
        self.buses.pop(session_id, None)
        await self.repo.delete_session(session_id)
        if self.checkpointer is not None and hasattr(self.checkpointer, "adelete_thread"):
            try:
                await self.checkpointer.adelete_thread(session_id)
            except Exception:  # noqa: BLE001
                log.warning("could not delete checkpoints for %s", session_id, exc_info=True)

    async def shutdown(self) -> None:
        for h in list(self.runs.values()):
            h.task.cancel()
        await asyncio.gather(*(h.task for h in self.runs.values()), return_exceptions=True)
        await self.mcp.close_all()

    # ------------------------------------------------------------------ the run
    async def _execute(self, session_id: str, run_id: str, text: str, cfg: dict[str, Any], session: dict[str, Any]) -> None:
        bus = self.bus(session_id)
        rec = RunRecorder(bus, self.repo, session_id, run_id)
        status, error = "completed", None
        final_answer = ""
        try:
            await rec.emit("run_started", mode=cfg.get("mode"), thinking=cfg.get("thinking", True))
            sandbox = self.sandboxes.get(session_id)
            await rec.emit("status", text="Starting sandbox", agent="main")
            await sandbox.ensure()
            await rec.emit("sandbox", status=(await sandbox.status()).to_dict())

            store = DbSessionStore(self.repo, session_id)
            core = await store.get_core_memory()
            approvals = ApprovalManager(bus, run_id)
            self.runs[session_id].approvals = approvals
            ctx = ToolContext(
                session_id=session_id, sandbox=sandbox, settings=self.settings, store=store,
                memory=self.memory if cfg.get("memory", True) else None, events=rec, approvals=approvals, search=self.search,
                approval_mode=cfg.get("approval", "auto"), agent="main",
            )

            mcp_tools, mcp_errors = [], {}
            selected, custom = list(cfg.get("mcp", [])), list(cfg.get("custom_mcp", []))
            if selected or custom:
                await rec.emit("status", text="Connecting MCP servers", agent="main")
                mcp_tools, mcp_errors = await self.mcp.ensure(session_id, sandbox, selected, custom)
                warned = self.mcp_warned.setdefault(session_id, set())
                for name, err in mcp_errors.items():
                    if (name, err) not in warned:  # tell the user once per distinct failure
                        warned.add((name, err))
                        await rec.emit("notice", level="warn", agent="main", text=f"MCP server '{name}' unavailable: {err}")
                await rec.emit("mcp", servers=self.mcp.status(session_id), tools=[t.name for t in mcp_tools])

            model = self.make_model(enable_thinking=bool(cfg.get("thinking", True)))
            env = RunEnv(
                settings=self.settings, model=model, recorder=rec, ctx=ctx, repo=self.repo, mcp_tools=mcp_tools,
                thinking=bool(cfg.get("thinking", True)), memory_enabled=bool(cfg.get("memory", True)), core_memory=core,
                options={"team_parallelism": cfg.get("team_parallelism"), "team_review": cfg.get("team_review", True)},
            )
            callbacks = [RunCallbacks(rec)]
            base_cfg = {"callbacks": callbacks}
            ctx.delegate = make_delegate(env, lambda: base_cfg)

            # --- recall -------------------------------------------------------------------
            recalled = ""
            if ctx.memory is not None:
                items = await ctx.memory.search(text, session_id=session_id, limit=self.settings.memory_recall_k)
                if items:
                    recalled = "\n\n[Relevant memories from earlier - use them if helpful]\n" + "\n".join(f"- ({m['kind']}) {m['content']}" for m in items)
                    await rec.emit("memory", action="recall", count=len(items), items=[{"id": m["id"], "kind": m["kind"], "content": m["content"]} for m in items])

            # --- mode ---------------------------------------------------------------------
            mode = cfg.get("mode", "auto")
            if mode == "auto":
                mode, reason = await self._triage(model, text, base_cfg)
                await rec.emit("mode_selected", mode=mode, reason=reason)
            self.runs[session_id].mode = mode

            agent_tools = build_tools(ctx, GENERALIST.tool_groups if ctx.memory is not None else tuple(g for g in GENERALIST.tool_groups if g != "memory"),
                                      extra=mcp_tools, max_tools=30)
            deps = AgentDeps(
                settings=self.settings, model=model, recorder=rec, ctx=ctx, tools=agent_tools,
                system_prompt=build_system_prompt(agent="Decillion", workdir=sandbox.workdir, role=None, core_memory=core,
                                                  memory_enabled=ctx.memory is not None),
                agent_name="main", max_steps=self.settings.max_steps, thinking=bool(cfg.get("thinking", True)),
            )
            graph = build_agent_graph(deps, self.checkpointer)
            thread = {"configurable": {"thread_id": session_id}, "callbacks": callbacks, "recursion_limit": self.settings.max_steps * 3 + 15}

            if mode == "team":
                team = build_team_graph(env, lambda: base_cfg)
                out = await team.ainvoke({"goal": text}, {**base_cfg, "recursion_limit": 120})
                final_answer = out.get("final", "")
                # keep the main thread coherent for follow-up questions
                await graph.aupdate_state(thread, {"messages": [HumanMessage(content=text), AIMessage(content=final_answer or "(team finished)")]})
            else:
                payload = {"messages": [HumanMessage(content=text + recalled)], "steps": 0, "nudges": 0, "format_errors": 0, "empties": 0, "sigs": []}
                token = current_agent.set("main")
                try:
                    out = await graph.ainvoke(payload, thread)
                finally:
                    current_agent.reset(token)
                final_answer = final_text(out["messages"])
                await self._reflect(model, ctx, out["messages"], rec, cfg)
            await self._maybe_title(session, text)
        except asyncio.CancelledError:
            status = "cancelled"
            await rec.emit("notice", level="warn", agent="main", text="Run cancelled.")
        except BackendError as exc:
            status, error = "failed", str(exc)
            await rec.emit("error", message=f"Model server problem: {exc}")
        except Exception as exc:  # noqa: BLE001
            log.exception("run %s failed", run_id)
            status, error = "failed", f"{type(exc).__name__}: {exc}"
            await rec.emit("error", message=error)
        finally:
            await self._finish(session_id, run_id, status, error, rec)

    async def _finish(self, session_id: str, run_id: str, status: str, error: str | None, rec: RunRecorder) -> None:
        try:
            await self.repo.finish_run(run_id, status, error, rec.usage)
            s = await self.repo.get_session(session_id)
            if s:
                total = dict(s.get("usage") or {})
                for k, v in rec.usage.items():
                    total[k] = total.get(k, 0) + v
                await self.repo.set_fields(session_id, usage=total)
        finally:
            await rec.emit("run_finished", status=status, error=error, usage=rec.usage)
            h = self.runs.get(session_id)
            if h and h.run_id == run_id:
                h.approvals = None

    async def _triage(self, model: A1ChatModel, text: str, cfg: dict[str, Any]) -> tuple[str, str]:
        if TEAM_HINT.search(text):
            return "team", "the request asks for a team/multi-agent approach"
        if len(text) < 140 and "\n" not in text:
            return "agent", "short request"
        try:
            t = await model.aextract(
                [SystemMessage(content=TRIAGE_PROMPT), HumanMessage(content=text[:3000])], Triage, name="classify_request", max_tokens=120,
            )
            if t.complexity.lower().startswith("complex") and t.parallelizable:  # type: ignore[attr-defined]
                return "team", "complex, parallelizable request"
            return "agent", f"{t.complexity} request"  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            return "agent", "triage unavailable"

    async def _reflect(self, model: A1ChatModel, ctx: ToolContext, messages: list, rec: RunRecorder, cfg: dict[str, Any]) -> None:
        """Post-run memory consolidation: extract durable facts/lessons (best effort)."""
        if ctx.memory is None or rec.usage["tool_calls"] < 3:
            return
        try:
            tx = truncate_middle(transcript(messages, tool_chars=300, ai_chars=500), 7000)
            res = await model.aextract([HumanMessage(content=REFLECT_PROMPT.format(transcript=tx))], MemoryExtraction, name="save_memories", max_tokens=500)
            saved = []
            for m in res.memories[:4]:  # type: ignore[attr-defined]
                if len(m.content.strip()) < 12:
                    continue
                saved.append(await ctx.memory.save(session_id=ctx.session_id, kind=m.kind, content=m.content, importance=m.importance, source="reflection"))
            if saved:
                await rec.emit("memory", action="reflect", count=len(saved), items=[{"id": s["id"], "kind": s["kind"], "content": s["content"]} for s in saved])
        except Exception:  # noqa: BLE001
            log.info("reflection skipped", exc_info=True)

    async def _maybe_title(self, session: dict[str, Any], text: str) -> None:
        if session.get("title") in (None, "", "New session"):
            first = re.sub(r"\s+", " ", text).strip()
            title = first[:60] + ("…" if len(first) > 60 else "")
            await self.repo.update_session(session["id"], title=title)
            await self.bus(session["id"]).emit("session_updated", title=title)
