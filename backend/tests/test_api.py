import asyncio

import httpx
import pytest

from app.config import Settings
from app.main import create_app

from .conftest import TEST_DB, ScriptedBackend, needs_pg
from .helpers import turn

pytestmark = needs_pg


def settings(tmp_path, **kw) -> Settings:
    return Settings(database_url=TEST_DB, sandbox_backend="local", sandbox_local_root=str(tmp_path / "sbx"),
                    llm_base_url="http://unused", enable_web_tools=False, **kw)


def agent_script(req, n):
    p = req.prompt
    if "Extract" in p or p.endswith("<function=save_memories>\n"):
        return '<parameter=memories>\n[{"content": "The project uses hello.py as its entry point.", "kind": "project", "importance": 3}]\n</parameter>\n</function>\n</tool_call>'
    if "<tool_response>" not in p:
        return turn("", write_file={"path": "hello.py", "content": "print('hi from agent')\n"}) 
    if p.count("<tool_response>") == 1:
        return turn("", bash={"command": "python3 hello.py"}) + "\n" + ""
    if p.count("<tool_response>") == 2:
        return turn("", show_table={"title": "Result", "columns": ["file", "output"], "rows": [["hello.py", "hi from agent"]]})
    return turn("Created hello.py and ran it: it prints 'hi from agent'.")


@pytest.fixture
async def app_client(tmp_path):
    from sqlalchemy import text

    from app.db.models import Base
    from app.db.session import Database

    s = settings(tmp_path)
    tmp = Database(s)
    async with tmp.engine.begin() as conn:
        await conn.execute(text("DROP TABLE IF EXISTS checkpoint_writes, checkpoint_blobs, checkpoints, checkpoint_migrations CASCADE"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        await conn.run_sync(Base.metadata.drop_all)
    await tmp.dispose()
    app = create_app(s)
    async with app.router.lifespan_context(app):
        c = app.state.container
        be = ScriptedBackend(fn=agent_script)
        c.backend = be
        c.service.backend = be
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            yield client, c, be


async def wait_idle(c, sid, timeout=30):
    for _ in range(int(timeout * 20)):
        if not c.service.is_running(sid):
            return
        await asyncio.sleep(0.05)
    raise AssertionError("run did not finish")


async def test_full_run_through_rest(app_client):
    client, c, be = app_client
    assert (await client.get("/api/health")).json()["database"] is True
    sid = (await client.post("/api/sessions", json={"config": {"mode": "agent", "mcp": []}})).json()["id"]
    r = await client.post(f"/api/sessions/{sid}/messages", json={"content": "Create hello.py and run it"})
    assert r.status_code == 202
    assert (await client.post(f"/api/sessions/{sid}/messages", json={"content": "again"})).status_code == 409  # one run at a time
    await wait_idle(c, sid)

    msgs = (await client.get(f"/api/sessions/{sid}/messages")).json()
    roles = [m["role"] for m in msgs]
    assert roles[0] == "user" and "tool" in roles and "card" in roles and roles[-1] == "assistant"
    card = next(m for m in msgs if m["role"] == "card")["meta"]["card"]
    assert card["kind"] == "table" and card["data"]["rows"][0][0] == "hello.py"
    assert "hello.py" in msgs[-1]["content"]
    tool = next(m for m in msgs if m["role"] == "tool" and m["tool_name"] == "bash")
    assert "hi from agent" in tool["content"]

    # files visible through the API
    files = (await client.get(f"/api/sessions/{sid}/files")).json()
    assert [e["name"] for e in files["entries"]] == ["hello.py"]
    f = (await client.get(f"/api/sessions/{sid}/file", params={"path": "hello.py"})).json()
    assert "hi from agent" in f["content"]
    term = (await client.post(f"/api/sessions/{sid}/terminal", json={"command": "python3 hello.py"})).json()
    assert term["exit_code"] == 0 and "hi from agent" in term["stdout"]

    # session bookkeeping
    s = (await client.get(f"/api/sessions/{sid}")).json()
    assert s["title"].startswith("Create hello.py") and s["usage"]["llm_calls"] >= 4
    # long-term memory was reflected from the run and is retrievable
    mems = (await client.get("/api/memories", params={"session_id": sid})).json()
    assert any("hello.py" in m["content"] for m in mems)
    # model saw the native template with its tools
    assert "<function=show_table>" in be.requests[0].prompt or '"name": "show_table"' in be.requests[0].prompt


async def test_conversation_state_survives_in_postgres_checkpointer(app_client):
    client, c, be = app_client
    sid = (await client.post("/api/sessions", json={"config": {"mode": "agent", "mcp": [], "memory": False}})).json()["id"]
    await client.post(f"/api/sessions/{sid}/messages", json={"content": "Create hello.py and run it"})
    await wait_idle(c, sid)
    be.fn = lambda req, n: turn("Yes, it printed: hi from agent")
    await client.post(f"/api/sessions/{sid}/messages", json={"content": "What did it print?"})
    await wait_idle(c, sid)
    last_prompt = be.requests[-1].prompt
    assert "Create hello.py and run it" in last_prompt and "hi from agent" in last_prompt and "What did it print?" in last_prompt


async def test_approval_flow_over_rest(app_client):
    client, c, be = app_client
    be.fn = lambda req, n: (turn("", bash={"command": "touch made.txt"}) if "<tool_response>" not in req.prompt else turn("done"))
    sid = (await client.post("/api/sessions", json={"config": {"mode": "agent", "mcp": [], "approval": "ask", "memory": False}})).json()["id"]
    await client.post(f"/api/sessions/{sid}/messages", json={"content": "touch a file"})
    bus = c.service.bus(sid)
    for _ in range(200):
        pend = [e for e in bus.buffer if e["type"] == "approval_request"]
        if pend:
            break
        await asyncio.sleep(0.05)
    assert pend and pend[0]["tool"] == "bash"
    r = await client.post(f"/api/sessions/{sid}/approvals/{pend[0]['approval_id']}", json={"approve": False, "note": "no"})
    assert r.status_code == 200
    await wait_idle(c, sid)
    files = (await client.get(f"/api/sessions/{sid}/files")).json()["entries"]
    assert files == []


async def test_cancel_and_error_paths(app_client):
    client, c, be = app_client

    async def slow(req, n):
        await asyncio.sleep(0)

    class Slow(ScriptedBackend):
        async def stream(self, req):
            from app.llm.backends import StreamPiece
            yield StreamPiece(text="thinking ")
            await asyncio.sleep(30)

    c.service.backend = Slow()
    sid = (await client.post("/api/sessions", json={"config": {"mode": "agent", "mcp": [], "memory": False}})).json()["id"]
    await client.post(f"/api/sessions/{sid}/messages", json={"content": "do slow"})
    await asyncio.sleep(0.5)
    assert (await client.post(f"/api/sessions/{sid}/cancel")).json()["cancelled"] is True
    await wait_idle(c, sid)
    types = [e["type"] for e in c.service.bus(sid).buffer]
    assert types[-1] == "run_finished" and c.service.bus(sid).buffer[-1]["status"] == "cancelled"
    assert (await client.get("/api/sessions/does-not-exist")).status_code == 404


async def test_backend_down_reports_failure_cleanly(app_client):
    client, c, be = app_client
    from app.llm.backends import BackendError

    class Down(ScriptedBackend):
        async def stream(self, req):
            raise BackendError("Cannot reach Ollama at http://ollama:11434")
            yield

    c.service.backend = Down()
    sid = (await client.post("/api/sessions", json={"config": {"mode": "agent", "mcp": [], "memory": False}})).json()["id"]
    await client.post(f"/api/sessions/{sid}/messages", json={"content": "hi"})
    await wait_idle(c, sid)
    evs = c.service.bus(sid).buffer
    assert any(e["type"] == "error" and "Cannot reach" in e["message"] for e in evs) and evs[-1]["status"] == "failed"


async def test_session_delete_cleans_everything(app_client):
    client, c, be = app_client
    sid = (await client.post("/api/sessions", json={"config": {"mode": "agent", "mcp": [], "memory": False}})).json()["id"]
    await client.post(f"/api/sessions/{sid}/messages", json={"content": "Create hello.py and run it"})
    await wait_idle(c, sid)
    assert (await client.delete(f"/api/sessions/{sid}")).status_code == 204
    assert (await client.get(f"/api/sessions/{sid}")).status_code == 404
    assert not (await c.sandboxes.get(sid).exists("hello.py"))


async def test_auth_when_key_configured(tmp_path):
    s = settings(tmp_path, app_api_key="sekret")
    app = create_app(s)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            assert (await client.get("/api/health")).status_code == 200  # health is public
            assert (await client.get("/api/sessions")).status_code == 401
            assert (await client.get("/api/sessions", headers={"x-api-key": "wrong"})).status_code == 401
            assert (await client.get("/api/sessions", headers={"Authorization": "Bearer sekret"})).status_code == 200
            assert (await client.get("/api/sessions", params={"token": "sekret"})).status_code == 200


def test_websocket_streaming(tmp_path):
    from starlette.testclient import TestClient

    from sqlalchemy import create_engine, text
    from app.db.models import Base

    sync_url = TEST_DB.replace("+asyncpg", "+psycopg")
    eng = create_engine(sync_url)
    with eng.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS checkpoint_writes, checkpoint_blobs, checkpoints, checkpoint_migrations CASCADE"))
        Base.metadata.drop_all(conn)
    eng.dispose()

    app = create_app(settings(tmp_path))
    with TestClient(app) as tc:
        c = app.state.container
        c.backend = c.service.backend = ScriptedBackend(fn=lambda req, n: turn("hello ws") if True else "")
        sid = tc.post("/api/sessions", json={"config": {"mode": "agent", "mcp": [], "memory": False}}).json()["id"]
        with tc.websocket_connect(f"/api/sessions/{sid}/ws") as ws:
            assert ws.receive_json()["type"] == "hello"
            ws.send_json({"type": "send", "content": "hi there"})
            seen = []
            while True:
                evt = ws.receive_json()
                seen.append(evt["type"])
                if evt["type"] == "run_finished":
                    break
        assert "user_message" in seen and "reasoning_delta" in seen and "content_delta" in seen and "message_end" in seen
        # a second connection replays the buffered events (page reload)
        with tc.websocket_connect(f"/api/sessions/{sid}/ws?since=0") as ws2:
            assert ws2.receive_json()["type"] == "hello"
            assert ws2.receive_json()["type"] in {"run_started", "user_message"}


async def test_team_mode_run_and_delegation_have_no_duplicate_events(app_client):
    client, c, be = app_client
    from .test_team import scripted

    be.fn = scripted
    sid = (await client.post("/api/sessions", json={"config": {"mode": "team", "mcp": [], "memory": False}})).json()["id"]
    await client.post(f"/api/sessions/{sid}/messages", json={"content": "Build calc"})
    await wait_idle(c, sid, 60)
    evs = c.service.bus(sid).buffer
    assert evs[-1]["type"] == "run_finished" and evs[-1]["status"] == "completed", [e for e in evs if e["type"] == "error"]
    starts = [e["message_id"] for e in evs if e["type"] == "message_start"]
    ends = [e["message_id"] for e in evs if e["type"] == "message_end"]
    assert len(starts) == len(set(starts)) and set(ends) <= set(starts)  # no duplicated handlers
    tasks = (await client.get(f"/api/sessions/{sid}/tasks")).json()
    assert {t["key"]: t["status"] for t in tasks} == {"t1": "done", "t2": "done", "t3": "done"}
    msgs = (await client.get(f"/api/sessions/{sid}/messages")).json()
    assert msgs[-1]["role"] == "assistant" and "FINAL" in msgs[-1]["content"]
    # follow-up question continues from the team's result
    be.fn = lambda req, n: turn("It is in calc.py")
    await client.post(f"/api/sessions/{sid}/messages", json={"content": "Where is the code?", "options": {"mode": "agent"}})
    await wait_idle(c, sid)
    assert "FINAL: calc.py" in be.requests[-1].prompt


async def test_delegate_tool_events_not_duplicated(app_client):
    client, c, be = app_client

    def fn(req, n):
        p = req.prompt
        if "<tool_response>" in p and "Report from" in p:
            return turn("Delegation finished.")
        if "You are one member" in p:  # worker
            return turn("worker report: found 7")
        return turn("", delegate_task={"role": "researcher", "task": "find the number"})

    be.fn = fn
    sid = (await client.post("/api/sessions", json={"config": {"mode": "agent", "mcp": [], "memory": False}})).json()["id"]
    await client.post(f"/api/sessions/{sid}/messages", json={"content": "delegate please"})
    await wait_idle(c, sid)
    evs = c.service.bus(sid).buffer
    starts = [e["message_id"] for e in evs if e["type"] == "message_start"]
    assert len(starts) == len(set(starts)) == 3  # main, worker, main
    assert any(e["type"] == "delegation" for e in evs)
