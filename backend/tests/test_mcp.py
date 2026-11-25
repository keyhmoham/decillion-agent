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


async def test_failed_server_is_not_restarted_on_every_run(tmp_path, monkeypatch):
    s = Settings(sandbox_backend="local", sandbox_local_root=str(tmp_path), mcp_startup_timeout=10)
    sb = LocalSandbox("sess-mcp-0003", s)
    await sb.ensure()
    mgr = McpManager(s, backend_env={})
    custom = [{"name": "bad", "command": ["false"]}]  # exists (passes pre-flight) but exits immediately
    await mgr.ensure("sess-mcp-0003", sb, [], custom)
    first = mgr._handles["sess-mcp-0003"]["bad"]
    await mgr.ensure("sess-mcp-0003", sb, [], custom)
    assert mgr._handles["sess-mcp-0003"]["bad"] is first  # backoff: same handle, no new process attempt
    import app.mcp.manager as m
    monkeypatch.setattr(m, "RETRY_AFTER", -1)
    await mgr.ensure("sess-mcp-0003", sb, [], custom)
    assert mgr._handles["sess-mcp-0003"]["bad"] is not first  # retried after the back-off window
    await mgr.close_all()


# ----------------------------------------------------------------------------- pre-flight diagnostics
class _StubDockerSandbox:
    backend = "docker"
    workdir = "/workspace"

    def __init__(self, have: set[str]):
        self.have = have

    def mcp_stdio(self, argv, env=None):
        return "docker", ["exec", "-i", "c", *argv], {"PATH": "/usr/bin"}

    async def exec(self, command, **kw):
        from app.sandbox.base import ExecResult

        binary = command.split()[-1].strip("'")
        return ExecResult(0 if binary in self.have else 1)


async def test_missing_docker_cli_gives_actionable_error(monkeypatch):
    import app.mcp.manager as m

    monkeypatch.setattr(m.shutil, "which", lambda c: None)
    mgr = McpManager(Settings(sandbox_backend="docker"), backend_env={})
    tools, errors = await mgr.ensure("s1", _StubDockerSandbox(set()), ["time"])
    assert tools == [] and "docker' CLI is not installed in the backend container" in errors["time"]
    assert "docker compose build backend" in errors["time"]


async def test_missing_binary_in_sandbox_image_is_named(monkeypatch):
    import app.mcp.manager as m

    monkeypatch.setattr(m.shutil, "which", lambda c: "/usr/bin/docker")
    mgr = McpManager(Settings(sandbox_backend="docker", sandbox_image="decillion-sandbox:latest"), backend_env={})
    _, errors = await mgr.ensure("s2", _StubDockerSandbox({"mcp-server-git"}), ["time", "git"])
    assert "'mcp-server-time' is not installed in the sandbox image (decillion-sandbox:latest)" in errors["time"]
    assert "docker compose build sandbox-image" in errors["time"]
    assert "git" not in errors or "not installed" not in errors["git"]  # git's binary exists -> passes pre-flight


async def test_preflight_failures_are_rechecked_every_run(monkeypatch):
    import app.mcp.manager as m

    monkeypatch.setattr(m.shutil, "which", lambda c: "/usr/bin/docker")
    mgr = McpManager(Settings(sandbox_backend="docker"), backend_env={})
    sb = _StubDockerSandbox(set())
    _, e1 = await mgr.ensure("s3", sb, ["time"])
    assert "not installed in the sandbox image" in e1["time"]
    sb.have.add("mcp-server-time")  # operator rebuilt the image
    _, e2 = await mgr.ensure("s3", sb, ["time"])
    assert "not installed" not in e2.get("time", "")  # no 5-minute back-off for cheap pre-flight problems


def test_git_preset_bootstraps_a_repository_and_docker_exec_forwards_daemon_env(monkeypatch):
    from app.mcp.catalog import PRESETS
    from app.sandbox.docker_sandbox import DockerSandbox

    argv = PRESETS["git"].argv
    assert argv[0] == "bash" and "git init" in argv[2] and "exec mcp-server-git" in argv[2]
    monkeypatch.setenv("DOCKER_HOST", "tcp://socket-proxy:2375")
    sb = DockerSandbox("abcd1234-0000-0000-0000-000000000000", Settings(sandbox_backend="docker"), client=object())
    cmd, args, env = sb.mcp_stdio(list(argv))
    assert cmd == "docker" and env["DOCKER_HOST"] == "tcp://socket-proxy:2375" and "PATH" in env
    assert args[:2] == ["exec", "-i"]
