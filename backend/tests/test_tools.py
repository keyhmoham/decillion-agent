import asyncio

import pytest

from app.tools.registry import build_tools

from .helpers import make_ctx


@pytest.fixture
def toolbox(tmp_path):
    ctx, bus, rec, s = make_ctx(tmp_path)

    class FakeMem:
        def __init__(self):
            self.items = []

        async def save(self, **kw):
            self.items.append(kw)
            return {"id": "x" * 36, "deduplicated": False, **kw}

        async def search(self, q, **kw):
            return []

        async def forget(self, mid):
            return mid == "x" * 36

    ctx.memory = FakeMem()
    tools = {t.name: t for t in build_tools(ctx, ("sandbox", "plan", "present", "memory"))}
    return ctx, bus, tools


async def call(tools, name, **args):
    return await tools[name].ainvoke(args)


async def test_write_read_edit_cycle(toolbox):
    ctx, bus, t = toolbox
    out = await call(t, "write_file", path="src/app.py", content="a = 1\nb = 2\nb = 2\n")
    assert "Created" in out and "3 lines" in out
    r = await call(t, "read_file", path="src/app.py")
    assert "     1\ta = 1" in r and "(3 lines)" in r
    assert "occurs 2 times" in await call(t, "edit_file", path="src/app.py", old_string="b = 2", new_string="b = 3")
    assert "not found" in (await call(t, "edit_file", path="src/app.py", old_string="a  =  1", new_string="a = 9")).lower()
    ok = await call(t, "edit_file", path="src/app.py", old_string="a = 1", new_string="a = 9")
    assert "Edited" in ok and "+a = 9" in ok
    assert "2 replacements" in await call(t, "edit_file", path="src/app.py", old_string="b = 2", new_string="b = 3", replace_all=True)
    assert "a = 9" in await call(t, "read_file", path="src/app.py", offset=1, limit=1)
    assert "Error: File not found" in await call(t, "read_file", path="nope.txt")
    assert any(e["type"] == "file_changed" for e in bus.buffer)


async def test_bash_exit_codes_stderr_timeout(toolbox):
    ctx, _, t = toolbox
    ok = await call(t, "bash", command="echo hi; echo oops >&2; exit 3")
    assert "hi" in ok and "[stderr]" in ok and "oops" in ok and "[exit code 3]" in ok
    slow = await call(t, "bash", command="sleep 5", timeout=1)
    assert "TIMED OUT" in slow


async def test_bash_streams_output_events(toolbox):
    ctx, bus, t = toolbox
    await call(t, "bash", command="echo one; sleep 0.2; echo two")
    chunks = "".join(e["text"] for e in bus.buffer if e["type"] == "tool_output")
    assert "one" in chunks and "two" in chunks


async def test_background_job_lifecycle(toolbox):
    ctx, _, t = toolbox
    started = await call(t, "bash", command="echo booted; sleep 30", background=True)
    job = started.split("job ")[1].split(".")[0]
    log = await call(t, "bash_job", job_id=job)
    assert "booted" in log and "still running" in log
    assert "Sent SIGTERM" in await call(t, "bash_job", job_id=job, action="kill")
    await asyncio.sleep(0.5)
    assert "exited" in await call(t, "bash_job", job_id=job)
    assert "Invalid job id" in await call(t, "bash_job", job_id="../../etc")


async def test_list_glob_grep(toolbox):
    ctx, _, t = toolbox
    await call(t, "write_file", path="pkg/a.py", content="def alpha():\n    return 'needle'\n")
    await call(t, "write_file", path="pkg/b.txt", content="nothing\n")
    tree = await call(t, "list_dir", path=".", depth=2)
    assert "pkg/" in tree and "pkg/a.py" in tree
    assert "pkg/a.py" in await call(t, "glob", pattern="*.py")
    g = await call(t, "grep", pattern="needle", path=".")
    assert "a.py" in g and "needle" in g
    assert "(no matches)" in await call(t, "grep", pattern="zzzz")


async def test_todo_tools_enforce_single_in_progress(toolbox):
    ctx, bus, t = toolbox
    out = await call(t, "write_todos", todos=[{"content": "a", "status": "in_progress"}, {"content": "b", "status": "in_progress"}, {"content": "c"}])
    assert [x["status"] for x in ctx.store.todos] == ["in_progress", "pending", "pending"] and "0/3" in out
    out = await call(t, "update_todo", index=1, status="completed")
    assert ctx.store.todos[1]["status"] == "in_progress" and "Next open step: 2" in out
    assert "between 1 and 3" in await call(t, "update_todo", index=9, status="completed")


async def test_card_tools_validate_and_emit(toolbox):
    ctx, bus, t = toolbox
    ok = await call(t, "show_chart", title="Latency", type="bar", labels=["a", "b"], series=[{"name": "p50", "values": [1, 2]}])
    assert "displayed" in ok
    card = [e for e in bus.buffer if e["type"] == "card"][-1]["card"]
    assert card["kind"] == "chart" and card["data"]["series"][0]["values"] == [1.0, 2.0]
    assert "must match" in await call(t, "show_chart", title="x", type="line", labels=["a", "b"], series=[{"name": "s", "values": [1]}])
    assert "displayed" in await call(t, "show_table", title="T", columns=["a", "b"], rows=[[1], [1, 2, 3]])
    rows = [e for e in bus.buffer if e["type"] == "card"][-1]["card"]["data"]["rows"]
    assert rows == [[1, None], [1, 2]]  # ragged rows are normalised
    assert "displayed" in await call(t, "show_stats", title="KPIs", items=[{"label": "Users", "value": "42", "delta": "+3", "status": "good"}])
    assert "displayed" in await call(t, "show_diagram", title="Flow", mermaid="graph TD; A-->B")
    await call(t, "write_file", path="plot.svg", content="<svg xmlns='http://www.w3.org/2000/svg'/>")
    assert "displayed" in await call(t, "show_image", path="plot.svg")
    assert "not found" in (await call(t, "show_image", path="missing.png")).lower()


async def test_memory_tools(toolbox):
    ctx, bus, t = toolbox
    assert "Saved memory" in await call(t, "memory_save", content="User likes dark mode in every app.", kind="preference", scope="global")
    assert ctx.memory.items[0]["session_id"] is None
    assert "No matching" in await call(t, "memory_search", query="dark")
    assert "updated" in await call(t, "core_memory_update", block="user", content="Name: Sam")
    assert ctx.store.core["user"] == "Name: Sam"
    assert "Too long" in await call(t, "core_memory_update", block="notes", content="x" * 2000)
    assert "Memory deleted" in await call(t, "memory_forget", memory_id="x" * 36)


async def test_mcp_tools_are_gated_unless_declared_read_only(tmp_path):
    from langchain_core.tools import StructuredTool

    ctx, *_ = make_ctx(tmp_path)

    async def noop(**kw):
        return "ok"

    ro = StructuredTool.from_function(coroutine=noop, name="get_thing", description="d", metadata={"readOnlyHint": True})
    rw = StructuredTool.from_function(coroutine=noop, name="git_commit", description="d", metadata={"readOnlyHint": False})
    unknown = StructuredTool.from_function(coroutine=noop, name="mystery", description="d")
    tools = {t.name: t for t in build_tools(ctx, ("readonly",), extra=[ro, rw, unknown])}
    assert tools["get_thing"].metadata["mutating"] is False
    assert tools["git_commit"].metadata["mutating"] is True and tools["mystery"].metadata["mutating"] is True
