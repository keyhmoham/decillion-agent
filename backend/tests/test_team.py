import json


from app.agents.env import RunEnv
from app.agents.team import PlanTask, TaskPlan, build_team_graph, normalise_plan, ready_tasks
from app.agents.workers import make_delegate

from .helpers import make_ctx, turn

PLAN_XML = (
    "<parameter=summary>\nBuild and verify\n</parameter>\n<parameter=tasks>\n"
    + json.dumps([
        {"key": "t1", "title": "Write module", "description": "Create calc.py with add(a,b)", "role": "coder", "depends_on": []},
        {"key": "t2", "title": "Write notes", "description": "Create NOTES.md", "role": "writer", "depends_on": []},
        {"key": "t3", "title": "Verify", "description": "Run calc.py", "role": "tester", "depends_on": ["t1", "t2"]},
    ])
    + "\n</parameter>\n</function>\n</tool_call>"
)


def scripted(req, n):
    p = req.prompt
    if p.endswith("<function=submit_plan>\n"):
        return PLAN_XML
    if p.endswith("<function=submit_verdict>\n"):
        return "<parameter=accepted>\ntrue\n</parameter>\n<parameter=feedback>\n\n</parameter>\n</function>\n</tool_call>"
    if "Write a compact plan" in p:
        return "think\n</think>\n\nt1 coder; t2 writer; t3 tester after both"
    if "The team finished working" in p:
        return "FINAL: calc.py and NOTES.md are in the workspace and verified."
    if "Your task [t1]" in p:
        if "<tool_response>" not in p:
            return turn("", write_file={"path": "calc.py", "content": "def add(a, b):\n    return a + b\nprint(add(2, 3))\n"})
        return turn("Wrote calc.py")
    if "Your task [t2]" in p:
        if "<tool_response>" not in p:
            return turn("", write_file={"path": "NOTES.md", "content": "# Notes\n"})
        return turn("Wrote NOTES.md")
    if "Your task [t3]" in p:
        if "<tool_response>" not in p:
            return turn("", bash={"command": "python3 calc.py"})
        return turn("Verified: prints 5")
    return turn("??")


async def test_team_end_to_end(make_model, tmp_path):
    ctx, bus, rec, settings = make_ctx(tmp_path, team_parallelism=2)
    model, be = make_model(fn=scripted, empty_retries=0)
    env = RunEnv(settings=settings, model=model, recorder=rec, ctx=ctx, options={"team_parallelism": 2})
    graph = build_team_graph(env)
    out = await graph.ainvoke({"goal": "Build calc"}, {"recursion_limit": 60})
    tasks = out["tasks"]
    assert {k: t["status"] for k, t in tasks.items()} == {"t1": "done", "t2": "done", "t3": "done"}
    assert "FINAL" in out["final"]
    assert await ctx.sandbox.exists("calc.py") and await ctx.sandbox.exists("NOTES.md")
    assert "5" in tasks["t3"]["result"]
    # t3 started only after both dependencies finished
    assert tasks["t3"]["started"] >= max(tasks["t1"]["finished"], tasks["t2"]["finished"])
    # t1 and t2 were dispatched in the same superstep (parallel): both started before either finished
    assert tasks["t2"]["started"] < tasks["t1"]["finished"] or tasks["t1"]["started"] < tasks["t2"]["finished"]
    agents = {e.get("agent") for e in bus.buffer if e["type"] == "tool_start"}
    assert {"coder:t1", "writer:t2", "tester:t3"} <= agents
    assert any(e["type"] == "plan" for e in bus.buffer)


async def test_failed_review_retries_with_feedback_then_accepts(make_model, tmp_path):
    ctx, bus, rec, settings = make_ctx(tmp_path)
    verdicts = {"n": 0}

    def fn(req, n):
        p = req.prompt
        if p.endswith("<function=submit_plan>\n"):
            return ("<parameter=summary>\ns\n</parameter>\n<parameter=tasks>\n"
                    + json.dumps([{"key": "t1", "title": "X", "description": "do x", "role": "coder", "depends_on": []}])
                    + "\n</parameter>\n</function>\n</tool_call>")
        if p.endswith("<function=submit_verdict>\n"):
            verdicts["n"] += 1
            ok = "true" if verdicts["n"] > 1 else "false"
            return f"<parameter=accepted>\n{ok}\n</parameter>\n<parameter=feedback>\nadd tests\n</parameter>\n</function>\n</tool_call>"
        if "Write a compact plan" in p:
            return "t\n</think>\n\nplan"
        if "The team finished working" in p:
            return "done report"
        return turn("worked")

    model, _ = make_model(fn=fn, empty_retries=0)
    env = RunEnv(settings=settings, model=model, recorder=rec, ctx=ctx)
    out = await build_team_graph(env).ainvoke({"goal": "g"}, {"recursion_limit": 60})
    t1 = out["tasks"]["t1"]
    assert t1["status"] == "done" and t1["attempts"] == 2
    assert any("review rejected" in e.get("text", "") for e in bus.buffer if e["type"] == "notice")


async def test_failed_dependency_blocks_dependents(make_model, tmp_path):
    ctx, bus, rec, settings = make_ctx(tmp_path, max_task_retries=0)

    def fn(req, n):
        p = req.prompt
        if p.endswith("<function=submit_plan>\n"):
            return ("<parameter=summary>\ns\n</parameter>\n<parameter=tasks>\n" + json.dumps([
                {"key": "t1", "title": "A", "description": "a", "role": "coder", "depends_on": []},
                {"key": "t2", "title": "B", "description": "b", "role": "tester", "depends_on": ["t1"]}]) + "\n</parameter>\n</function>\n</tool_call>")
        if p.endswith("<function=submit_verdict>\n"):
            return "<parameter=accepted>\nfalse\n</parameter>\n<parameter=feedback>\nbad\n</parameter>\n</function>\n</tool_call>"
        if "Write a compact plan" in p:
            return "t\n</think>\n\nplan"
        if "The team finished working" in p:
            assert "blocked" in p
            return "partial report"
        return turn("w")

    model, _ = make_model(fn=fn, empty_retries=0)
    env = RunEnv(settings=settings, model=model, recorder=rec, ctx=ctx)
    out = await build_team_graph(env).ainvoke({"goal": "g"}, {"recursion_limit": 60})
    assert out["tasks"]["t1"]["status"] == "failed" and out["tasks"]["t2"]["status"] == "blocked"
    assert out["final"] == "partial report"


async def test_unstructured_plan_falls_back_to_single_task(make_model, tmp_path):
    ctx, bus, rec, settings = make_ctx(tmp_path)

    def fn(req, n):
        p = req.prompt
        if p.endswith("<function=submit_plan>\n"):
            return "garbage that is not parameters"
        if "Write a compact plan" in p:
            return "t\n</think>\n\nplan"
        if "The team finished working" in p:
            return "ok"
        return turn("handled")

    model, _ = make_model(fn=fn, empty_retries=0)
    env = RunEnv(settings=settings, model=model, recorder=rec, ctx=ctx)
    out = await build_team_graph(env).ainvoke({"goal": "do things"}, {"recursion_limit": 60})
    assert list(out["tasks"]) == ["t1"] and out["tasks"]["t1"]["status"] == "done"


async def test_handoff_creates_follow_up_task(make_model, tmp_path):
    ctx, bus, rec, settings = make_ctx(tmp_path)

    def fn(req, n):
        p = req.prompt
        if p.endswith("<function=submit_plan>\n"):
            return ("<parameter=summary>\ns\n</parameter>\n<parameter=tasks>\n" + json.dumps([
                {"key": "t1", "title": "A", "description": "a", "role": "researcher", "depends_on": []}]) + "\n</parameter>\n</function>\n</tool_call>")
        if "Write a compact plan" in p:
            return "t\n</think>\n\nplan"
        if "The team finished working" in p:
            return "report"
        if "Your task [t1]" in p and "<tool_response>" not in p:
            return turn("", handoff={"role": "coder", "reason": "needs implementation"})
        return turn("summary of work")

    model, _ = make_model(fn=fn, empty_retries=0)
    env = RunEnv(settings=settings, model=model, recorder=rec, ctx=ctx)
    out = await build_team_graph(env).ainvoke({"goal": "g"}, {"recursion_limit": 60})
    assert "t1h1" in out["tasks"] and out["tasks"]["t1h1"]["role"] == "coder" and out["tasks"]["t1h1"]["status"] == "done"


async def test_delegate_tool_runs_sub_agent(make_model, tmp_path):
    ctx, bus, rec, settings = make_ctx(tmp_path)

    def fn(req, n):
        return turn("Delegated work done: 42") if "<tool_response>" in req.prompt or True else ""

    model, _ = make_model(fn=fn, empty_retries=0)
    env = RunEnv(settings=settings, model=model, recorder=rec, ctx=ctx)
    report = await make_delegate(env)("researcher", "find the answer")
    assert "Delegated work done: 42" in report and report.startswith("Report from researcher-1")
    assert any(e["type"] == "delegation" for e in bus.buffer)


def test_plan_normalisation_repairs_bad_plans():
    plan = TaskPlan(summary="s", tasks=[
        PlanTask(key="a", title="A", description="d", role="wizard", depends_on=["b", "zzz", "a"]),
        PlanTask(key="b", title="B", description="d", role="tester", depends_on=["a"]),   # cycle a<->b
        PlanTask(key="a", title="dup", description="d", role="coder", depends_on=[]),     # duplicate key
    ])
    tasks = normalise_plan(plan, "goal")
    keys = [t["key"] for t in tasks]
    assert len(set(keys)) == 3 and tasks[0]["role"] == "coder"
    assert ready_tasks({t["key"]: t for t in tasks})  # acyclic: something is runnable
