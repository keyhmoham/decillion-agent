"""Team orchestration: plan -> dispatch (parallel, dependency-aware) -> review/retry -> hand-offs -> synthesis."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from ..tools.base import truncate_middle
from .env import RunEnv
from .prompts import ORCHESTRATOR, REVIEW_PROMPT, SYNTH_PROMPT
from .roles import ROLE_NAMES, roles_for_prompt
from .workers import run_worker

log = logging.getLogger(__name__)
MAX_TASKS = 8
REVIEWED_ROLES = {"coder", "writer", "devops"}


# ----------------------------------------------------------------------------- structured outputs
class PlanTask(BaseModel):
    key: str = Field(description="Short id: t1, t2, ...")
    title: str = Field(description="Short title.")
    description: str = Field(description="Complete self-contained instructions, incl. file paths and acceptance criteria.")
    role: str = Field(description=f"One of: {', '.join(ROLE_NAMES)}.")
    depends_on: list[str] = Field(default_factory=list, description="Keys of tasks that must finish first.")


class TaskPlan(BaseModel):
    """The task plan for the team."""

    summary: str = Field(description="One-sentence description of the approach.")
    tasks: list[PlanTask] = Field(description="2-8 tasks.")


class Verdict(BaseModel):
    """Acceptance verdict for a task result."""

    accepted: bool = Field(description="true only if the task is clearly completed.")
    feedback: str = Field("", description="What is missing, if not accepted.")


# ----------------------------------------------------------------------------- plan hygiene
def normalise_plan(plan: TaskPlan, goal: str) -> list[dict[str, Any]]:
    """Make an LLM-written plan safe: valid roles, unique keys, existing deps, acyclic, bounded."""
    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, t in enumerate(plan.tasks[:MAX_TASKS], 1):
        key = (t.key or f"t{i}").strip().lower().replace(" ", "")
        if key in seen or not key:
            key = f"t{i}"
        seen.add(key)
        role = t.role.strip().lower()
        tasks.append({
            "key": key, "title": t.title.strip()[:200] or f"Task {i}", "description": t.description.strip() or t.title,
            "role": role if role in ROLE_NAMES else "coder", "depends_on": [d.strip().lower() for d in t.depends_on],
            "status": "pending", "result": "", "attempts": 0,
        })
    keys = {t["key"] for t in tasks}
    for t in tasks:
        t["depends_on"] = [d for d in dict.fromkeys(t["depends_on"]) if d in keys and d != t["key"]]
    # break cycles: drop deps that point "forward" in a cycle (Kahn)
    resolved: set[str] = set()
    remaining = {t["key"]: t for t in tasks}
    while remaining:
        ready = [k for k, t in remaining.items() if all(d in resolved for d in t["depends_on"])]
        if not ready:
            for t in remaining.values():  # cycle: cut all remaining dependency edges to unresolved nodes
                t["depends_on"] = [d for d in t["depends_on"] if d in resolved]
            continue
        for k in ready:
            resolved.add(k)
            remaining.pop(k)
    if not tasks:
        tasks = [{"key": "t1", "title": "Complete the request", "description": goal, "role": "coder", "depends_on": [],
                  "status": "pending", "result": "", "attempts": 0}]
    return tasks


def merge_tasks(a: dict[str, dict], b: dict[str, dict]) -> dict[str, dict]:
    out = {k: dict(v) for k, v in (a or {}).items()}
    for k, v in (b or {}).items():
        out[k] = {**out.get(k, {}), **v}
    return out


class TeamState(TypedDict, total=False):
    goal: str
    summary: str
    tasks: Annotated[dict[str, dict], merge_tasks]
    final: str


def ready_tasks(tasks: dict[str, dict]) -> list[dict]:
    return [
        t for t in tasks.values()
        if t["status"] == "pending" and all(tasks.get(d, {}).get("status") == "done" for d in t["depends_on"])
    ]


def build_team_graph(env: RunEnv, config_getter=lambda: None):
    s = env.settings
    rec = env.recorder
    model = env.model
    parallelism = int(env.options.get("team_parallelism") or s.team_parallelism)
    review_on = bool(env.options.get("team_review", True))

    async def publish(tasks: dict[str, dict]) -> None:
        await rec.emit("tasks", tasks=sorted(tasks.values(), key=lambda t: t["key"]))
        if env.repo is not None:
            for t in tasks.values():
                await env.repo.upsert_task(
                    env.ctx.session_id, rec.run_id, t["key"], title=t["title"], description=t["description"], role=t["role"],
                    status=t["status"], depends_on=t["depends_on"], result=t.get("result", ""), attempts=t.get("attempts", 0),
                    handoff_from=t.get("handoff_from"),
                    started_at=datetime.fromtimestamp(t["started"], timezone.utc) if t.get("started") else None,
                    finished_at=datetime.fromtimestamp(t["finished"], timezone.utc) if t.get("finished") else None,
                )

    def cfg(**meta: Any) -> RunnableConfig:
        base = config_getter() or {}
        return {**base, "metadata": {**(base.get("metadata") or {}), "agent": "orchestrator", **meta}}

    # ------------------------------------------------------------------ plan
    async def plan_node(state: TeamState) -> dict[str, Any]:
        goal = state["goal"]
        await rec.emit("status", text="Orchestrator is planning the team's work", agent="orchestrator")
        system = ORCHESTRATOR.format(roles=roles_for_prompt(), max_tasks=MAX_TASKS, workdir=env.ctx.sandbox.workdir)
        # 1) think about the decomposition (the model's strength) ...
        draft = await model.bind(enable_thinking=True, thinking_budget=1400, max_tokens=900).ainvoke(
            [SystemMessage(content=system), HumanMessage(content=f"Goal:\n{goal}\n\nWrite a compact plan: one line per task with role, what it produces and its dependencies.")],
            cfg(),
        )
        # 2) ... then extract it as strictly-typed data in the model's no-think mode.
        try:
            plan = await model.aextract(
                [SystemMessage(content=system), HumanMessage(content=f"Goal:\n{goal}\n\nDraft plan:\n{draft.content}\n\nNow output the final task plan.")],
                TaskPlan, name="submit_plan", max_tokens=1500,
            )
            tasks = normalise_plan(plan, goal)  # type: ignore[arg-type]
            summary = plan.summary  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            log.warning("plan extraction failed (%s); falling back to a single task", exc)
            tasks = normalise_plan(TaskPlan(summary="", tasks=[]), goal)
            summary = "Single-agent fallback plan."
            await rec.emit("notice", level="warn", agent="orchestrator", text="Could not structure a team plan; running as a single task.")
        tmap = {t["key"]: t for t in tasks}
        await rec.emit("plan", summary=summary, tasks=list(tmap.values()))
        await publish(tmap)
        return {"tasks": tmap, "summary": summary}

    # ------------------------------------------------------------------ dispatch / join
    async def join_node(state: TeamState) -> dict[str, Any]:
        tasks = state.get("tasks", {})
        updates: dict[str, dict] = {}
        changed = True
        view = {k: dict(v) for k, v in tasks.items()}
        while changed:  # propagate failures to dependents
            changed = False
            for k, t in view.items():
                if t["status"] == "pending" and any(view.get(d, {}).get("status") in ("failed", "blocked") for d in t["depends_on"]):
                    t["status"] = "blocked"
                    t["result"] = "Blocked: a prerequisite task failed."
                    updates[k] = {"status": "blocked", "result": t["result"]}
                    changed = True
        if updates:
            await publish(view)
        return {"tasks": updates} if updates else {}

    def dispatch(state: TeamState):
        tasks = state.get("tasks", {})
        ready = ready_tasks(tasks)[:parallelism]
        if not ready:
            return "synthesize"
        return [Send("worker", {"task": t, "goal": state["goal"], "tasks": tasks}) for t in ready]

    # ------------------------------------------------------------------ worker
    async def worker_node(payload: dict[str, Any]) -> dict[str, Any]:
        t: dict = dict(payload["task"])
        tasks: dict[str, dict] = payload["tasks"]
        key = t["key"]
        name = f"{t['role']}:{key}"
        t.update(status="running", attempts=t.get("attempts", 0) + 1, started=time.time())
        await rec.emit("task_update", task=t)
        await publish({**tasks, key: t})

        deps_text = "\n\n".join(
            f"[{d}] {tasks[d]['title']} ({tasks[d]['role']}) -> {truncate_middle(tasks[d].get('result', ''), 1600)}"
            for d in t["depends_on"] if d in tasks
        )
        prompt = (
            f"# Team goal\n{payload['goal']}\n\n# Your task [{key}]: {t['title']}\n{t['description']}\n"
            + (f"\n# Feedback from the previous attempt (fix this)\n{t['feedback']}\n" if t.get("feedback") else "")
            + (f"\n# Results of prerequisite tasks\n{deps_text}\n" if deps_text else "")
            + f"\n# Notes\n- Shared workspace: {env.ctx.sandbox.workdir}. Other agents' files are already there.\n"
              "- Do only your task. When done, reply with your report (files, verification, open issues)."
        )
        res = await run_worker(env, t["role"], prompt, agent_name=name, task_key=key, config=config_getter())
        out: dict[str, dict] = {}
        status = "done" if res.ok else "failed"
        feedback = ""
        if res.ok and review_on and t["role"] in REVIEWED_ROLES:
            try:
                verdict = await model.aextract(
                    [HumanMessage(content=REVIEW_PROMPT.format(title=t["title"], description=t["description"], result=truncate_middle(res.text, 3000)))],
                    Verdict, name="submit_verdict", max_tokens=300,
                )
                if not verdict.accepted:  # type: ignore[attr-defined]
                    feedback = verdict.feedback or "Result did not satisfy the task."  # type: ignore[attr-defined]
                    await rec.emit("notice", level="warn", agent="orchestrator", text=f"[{key}] review rejected the result: {feedback[:200]}")
            except Exception:  # noqa: BLE001
                log.warning("review of %s failed; accepting result", key, exc_info=True)
        if feedback:
            if t["attempts"] <= s.max_task_retries:
                status = "pending"
            else:
                status = "failed"
        t.update(status=status, result=res.text, feedback=feedback, finished=time.time() if status != "pending" else None)
        out[key] = t
        for i, h in enumerate(res.handoffs[:2], 1):  # swarm-style hand-off -> follow-up task
            hk = f"{key}h{i}"
            out[hk] = {
                "key": hk, "title": f"Continue {key}: {h['reason'][:80]}", "role": h["role"] if h["role"] in ROLE_NAMES else "coder",
                "description": f"Handed off from {name}. Reason: {h['reason']}\n\nThe previous agent's report:\n{res.text}\n\nOriginal task:\n{t['description']}",
                "depends_on": [key] if status == "done" else [], "status": "pending", "result": "", "attempts": 0, "handoff_from": key,
            }
        await rec.emit("task_update", task=t)
        await publish(merge_tasks(tasks, out))
        return {"tasks": out}

    # ------------------------------------------------------------------ synthesis
    async def synthesize_node(state: TeamState) -> dict[str, Any]:
        tasks = state.get("tasks", {})
        results = "\n\n".join(
            f"[{t['key']}] {t['title']} ({t['role']}, {t['status']})\n{truncate_middle(t.get('result', '') or '(no result)', 2200)}"
            for t in sorted(tasks.values(), key=lambda x: x["key"])
        )
        prompt = SYNTH_PROMPT.format(goal=state["goal"], results=results, workdir=env.ctx.sandbox.workdir)
        ai = await model.bind(enable_thinking=False, max_tokens=1500, temperature=0.4).ainvoke(
            [HumanMessage(content=prompt)], cfg(task_key=None)
        )
        return {"final": str(ai.content).strip()}

    g = StateGraph(TeamState)
    g.add_node("plan", plan_node)
    g.add_node("join", join_node)
    g.add_node("worker", worker_node)
    g.add_node("synthesize", synthesize_node)
    g.add_edge(START, "plan")
    g.add_edge("plan", "join")
    g.add_conditional_edges("join", dispatch, ["worker", "synthesize"])
    g.add_edge("worker", "join")
    g.add_edge("synthesize", END)
    return g.compile()
