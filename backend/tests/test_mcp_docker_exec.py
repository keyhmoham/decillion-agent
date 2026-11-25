"""MCP servers inside a Docker sandbox, through the SDK exec stream (no docker CLI).

A fake Docker "exec socket" runs a REAL MCP server as a subprocess and speaks Docker's actual
multiplexed stream framing (8-byte header: stream type, 3 zero bytes, big-endian length), so the
frame parsing, stdin writing, half-close and shutdown logic are exercised for real.
"""

import os
import shutil
import socket
import struct
import subprocess
import threading

import pytest

from app.config import Settings
from app.mcp.manager import McpManager

from .test_docker_sandbox import FakeClient, box, patch_docker_errors  # noqa: F401  (autouse fixture)

TIME_BIN = os.environ.get("TEST_MCP_TIME_BIN") or shutil.which("mcp-server-time")
pytestmark = pytest.mark.skipif(not TIME_BIN, reason="mcp-server-time not installed (set TEST_MCP_TIME_BIN)")


class SocketExecApi:
    """Docker low-level API surface used by DockerSandbox, with a real process behind exec sockets."""

    def __init__(self):
        self.created, self.procs = [], []
        self.code = 0

    def exec_create(self, cid, argv, **kw):
        self.created.append((argv, kw))
        return {"Id": f"ex{len(self.created)}"}

    def exec_start(self, eid, detach=False, tty=False, stream=False, socket=False, demux=False):
        argv, kw = self.created[int(eid[2:]) - 1]
        if not socket:  # plain command (pre-flight `command -v`): pretend it succeeded
            return iter([(b"/usr/local/bin/x\n", None)])
        client_side, server_side = _socketpair()
        env = {**os.environ, **{k: str(v) for k, v in (kw.get("environment") or {}).items()}}
        proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        self.procs.append(proc)

        def frame(kind, data):
            server_side.sendall(struct.pack(">BxxxL", kind, len(data)) + data)

        def pump_out(pipe, kind):
            for chunk in iter(lambda: pipe.read1(4096), b""):
                frame(kind, chunk)

        def pump_in():
            while True:
                data = server_side.recv(4096)
                if not data:  # client half-closed: docker closes the exec's stdin
                    proc.stdin.close()
                    return
                proc.stdin.write(data)
                proc.stdin.flush()

        def supervise():
            t1 = threading.Thread(target=pump_out, args=(proc.stdout, 1))
            t2 = threading.Thread(target=pump_out, args=(proc.stderr, 2))
            t1.start(); t2.start(); proc.wait(); t1.join(); t2.join()
            try:
                server_side.shutdown(socket_mod.SHUT_WR)
            except OSError:
                pass

        for fn in (pump_in, supervise):
            threading.Thread(target=fn, daemon=True).start()
        return socket_mod.SocketIO(client_side, "rwb")  # what docker-py returns for unix sockets

    def exec_inspect(self, eid):
        return {"ExitCode": self.code}


socket_mod = socket


def _socketpair():
    return socket_mod.socketpair()


def make_box(api):
    client = FakeClient()
    client.api = api
    sb, s = box(client)
    return sb, s


async def test_real_mcp_server_over_docker_exec_stream():
    api = SocketExecApi()
    sb, s = make_box(api)
    mgr = McpManager(Settings(sandbox_backend="docker", mcp_startup_timeout=30), backend_env={})
    tools, errors = await mgr.ensure("sess-x", sb, [], [{"name": "clock", "command": [TIME_BIN]}])
    try:
        assert not errors, errors
        assert {"get_current_time", "convert_time"} <= {t.name for t in tools}
        out = await next(t for t in tools if t.name == "get_current_time").ainvoke({"timezone": "Europe/Paris"})
        assert "Europe/Paris" in str(out)
        # the server really ran as an exec in the sandbox container, with stdin attached and our marker env
        argv, kw = next((a, k) for a, k in api.created if a == [TIME_BIN])
        assert kw["stdin"] is True and kw["tty"] is False and kw["user"] == "agent" and kw["workdir"] == "/workspace"
        assert "A1_MCP_ID" in kw["environment"]
    finally:
        import time

        t0 = time.monotonic()
        await mgr.close_all()
        assert time.monotonic() - t0 < 4  # half-close made the server exit; no force-kill needed
    assert all(p.poll() is not None for p in api.procs)  # no leaked processes


async def test_failing_server_reports_its_stderr():
    api = SocketExecApi()
    sb, _ = make_box(api)
    mgr = McpManager(Settings(sandbox_backend="docker", mcp_startup_timeout=15), backend_env={})
    script = "import sys; sys.stderr.write('fatal: not a git repository\\n'); sys.exit(2)"
    tools, errors = await mgr.ensure("sess-y", sb, [], [{"name": "broken", "command": ["python3", "-c", script]}])
    try:
        assert tools == [] and "broken" in errors
        assert "fatal: not a git repository" in errors["broken"]
    finally:
        await mgr.close_all()
