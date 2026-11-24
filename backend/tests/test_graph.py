import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agents.graph import AgentDeps, build_agent_graph, final_text, repair_dangling
from app.tools.registry import build_tools

from .helpers import call_text, make_ctx, turn


def _deps(make_model, tmp_path, scripts=None, fn=None, groups=("sandbox", "plan", "present"), **kw):
    ctx, bus, rec, settings = make_ctx(tmp_path, **kw.pop("settings", {}))
    model, be = make_model(scripts, fn=fn, empty_retries=0)
    tools = build_tools(ctx, groups)
    deps = AgentDeps(settings=settings, model=model, recorder=rec, ctx=ctx, tools=tools, system_prompt="SYS", max_steps=kw.pop("max_steps", 12), **kw)
    return deps, ctx, bus, be


async def run(deps, text="do it"):
    g = build_agent_graph(deps)
    return await g.ainvoke({"messages": [HumanMessage(content=text)]}, {"recursion_limit": 80})


async def test_write_run_and_finish(make_model, tmp_path):
    scripts = [
        turn("", write_file={"path": "hello.py", "content": "print(6*7)\n"}),
        turn("", bash={"command": "python3 hello.py"}),
        turn("Created hello.py; it prints 42."),
    ]
    deps, ctx, bus, be = _deps(make_model, tmp_path, scripts)
    out = await run(deps)
    assert "42" in final_text(out["messages"])
    tool_msgs = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert "Created" in tool_msgs[0].content and "42" in tool_msgs[1].content and "[exit code 0]" in tool_msgs[1].content
    # the tool result is fed back through the *native* template
    assert "<tool_response>" in be.requests[2].prompt
    types = [e["type"] for e in bus.buffer]
    assert "tool_start" in types and "tool_end" in types and "file_changed" in types


async def test_malformed_call_gets_format_help_then_recovers(make_model, tmp_path):
    scripts = [
        "hmm\n</think>\n\n<tool_call>\nnot valid at all\n</tool_call>",
        turn("done", think="retry"),
    ]
    deps, *_ = _deps(make_model, tmp_path, scripts)
    out = await run(deps)
    nudges = [m for m in out["messages"] if isinstance(m, ToolMessage) and m.name == "system"]
    assert nudges and "could not be parsed" in nudges[0].content
    assert final_text(out["messages"]) == "done"


async def test_unknown_tool_and_invalid_args_are_reported(make_model, tmp_path):
    scripts = [
        turn("", rm_rf={"path": "/"}),
        turn("", write_file={"path": "x.txt"}),  # missing content
        turn("gave up gracefully"),
    ]
    deps, *_ = _deps(make_model, tmp_path, scripts)
    out = await run(deps)
    tms = [m.content for m in out["messages"] if isinstance(m, ToolMessage)]
    assert "unknown tool 'rm_rf'" in tms[0] and "Available tools" in tms[0]
    assert "invalid arguments for write_file" in tms[1] and "content" in tms[1]


async def test_typed_arguments_reach_tools(make_model, tmp_path):
    scripts = [
        turn("", write_todos={"todos": [{"content": "a", "status": "pending"}, {"content": "b", "status": "pending"}]}),
        turn("", update_todo={"index": 1, "status": "completed"}),
        turn("", update_todo={"index": 2, "status": "completed"}),
        turn("all done"),
    ]
    deps, ctx, *_ = _deps(make_model, tmp_path, scripts)
    out = await run(deps)
    assert [t["status"] for t in ctx.store.todos] == ["completed", "completed"]
    assert final_text(out["messages"]) == "all done"


async def test_cannot_finish_with_open_todos_twice_then_allowed(make_model, tmp_path):
    scripts = [
        turn("", write_todos={"todos": [{"content": "step one", "status": "pending"}]}),
        turn("finished!"),           # tries to finish -> pushed back
        turn("still finished"),      # pushed back again
        turn("really finished"),     # allowed (nudge budget exhausted)
    ]
    deps, *_ = _deps(make_model, tmp_path, scripts)
    out = await run(deps)
    pushes = [m for m in out["messages"] if isinstance(m, ToolMessage) and "not marked completed" in m.content]
    assert len(pushes) == 2 and final_text(out["messages"]) == "really finished"


async def test_repeated_identical_calls_trigger_warning_then_finalize(make_model, tmp_path):
    def fn(req, n):
        if "out of steps" in req.prompt:
            return "I was stuck on a failing command."
        return turn("", bash={"command": "false"})

    deps, *_ = _deps(make_model, tmp_path, fn=fn, max_steps=30)
    out = await run(deps)
    warned = [m for m in out["messages"] if isinstance(m, ToolMessage) and "exact call" in m.content]
    assert warned
    assert "stuck" in final_text(out["messages"])


async def test_step_budget_finalizes_with_summary(make_model, tmp_path):
    def fn(req, n):
        if "out of steps" in req.prompt:
            return "Summary of progress."
        return turn("", bash={"command": f"echo {n}"})

    deps, *_ = _deps(make_model, tmp_path, fn=fn, max_steps=3)
    out = await run(deps)
    assert final_text(out["messages"]) == "Summary of progress."


async def test_approval_denied_blocks_mutating_tool(make_model, tmp_path):
    class Deny:
        async def request(self, **kw):
            return False, "not now"

    scripts = [turn("", bash={"command": "touch pwned"}), turn("ok I will not")]
    deps, ctx, *_ = _deps(make_model, tmp_path, scripts)
    ctx.approval_mode, ctx.approvals = "ask", Deny()
    out = await run(deps)
    assert not (await ctx.sandbox.exists("pwned"))
    assert any("user denied" in m.content for m in out["messages"] if isinstance(m, ToolMessage))


async def test_readonly_calls_may_run_in_parallel(make_model, tmp_path):
    scripts = [
        "x\n</think>\n\n" + call_text("list_dir", path=".") + "\n" + call_text("glob", pattern="*.md"),
        turn("done"),
    ]
    deps, *_ = _deps(make_model, tmp_path, scripts)
    out = await run(deps)
    assert len([m for m in out["messages"] if isinstance(m, ToolMessage)]) == 2


async def test_context_compaction_summarises_old_history(make_model, tmp_path):
    def fn(req, n):
        if "compacting an AI agent's working context" in req.prompt:
            return "## Goal\nBuild the thing\n## Done so far\nWrote files"
        if n < 6:
            return turn("", bash={"command": "python3 -c \"print('x'*3000)\""})
        return turn("final after compaction")

    deps, ctx, bus, be = _deps(make_model, tmp_path, fn=fn, settings={"context_budget": 1800})
    out = await run(deps)
    assert any(e["type"] == "compaction" for e in bus.buffer)
    msgs = out["messages"]
    assert any(isinstance(m, HumanMessage) and m.additional_kwargs.get("summary") for m in msgs)
    assert final_text(msgs) == "final after compaction"


def test_dangling_tool_calls_are_repaired():
    ai = AIMessage(content="", tool_calls=[{"name": "bash", "args": {}, "id": "c1"}])
    fixes = repair_dangling([HumanMessage(content="x"), ai])
    assert fixes and fixes[0].tool_call_id == "c1"
