import os
import shutil

import pytest

from app.config import Settings
from app.mcp.manager import McpManager
from app.sandbox.local_sandbox import LocalSandbox

TIME_BIN = os.environ.get("TEST_MCP_TIME_BIN") or shutil.which("mcp-server-time")
pytestmark = pytest.mark.skipif(not TIME_BIN, reason="mcp-server-time not installed (set TEST_MCP_TIME_BIN)")


async def test_custom_stdio_server_lifecycle(tmp_path):
    s = Settings(sandbox_backend="local", sandbox_local_root=str(tmp_path), mcp_startup_timeout=30)
    sb = LocalSandbox("sess-mcp-0001", s)
    await sb.ensure()
    mgr = McpManager(s, backend_env={})
    tools, errors = await mgr.ensure("sess-mcp-0001", sb, [], [{"name": "clock", "command": [TIME_BIN]}])
    try:
        assert not errors and {"get_current_time", "convert_time"} <= {t.name for t in tools}
        res = await next(t for t in tools if t.name == "get_current_time").ainvoke({"timezone": "UTC"})
        assert "UTC" in str(res)
        # idempotent: second ensure reuses the live process
        tools2, _ = await mgr.ensure("sess-mcp-0001", sb, [], [{"name": "clock", "command": [TIME_BIN]}])
        assert [t.name for t in tools2] == [t.name for t in tools]
        # deselect -> stopped
        tools3, _ = await mgr.ensure("sess-mcp-0001", sb, [], [])
        assert tools3 == []
    finally:
        await mgr.close_all()


async def test_broken_server_is_isolated(tmp_path):
    s = Settings(sandbox_backend="local", sandbox_local_root=str(tmp_path), mcp_startup_timeout=10)
    sb = LocalSandbox("sess-mcp-0002", s)
    await sb.ensure()
    mgr = McpManager(s, backend_env={})
    tools, errors = await mgr.ensure("sess-mcp-0002", sb, ["github"], [{"name": "bad", "command": ["/nonexistent/binary"]}])
    try:
        assert tools == [] and "bad" in errors and "github" in errors
    finally:
        await mgr.close_all()
