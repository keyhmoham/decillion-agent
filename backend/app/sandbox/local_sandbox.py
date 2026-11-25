"""Local-process sandbox.

!!! NOT ISOLATED !!!  Commands run directly on the backend host inside a per-session directory.
It exists so the whole agent stack (tools, graph, API) can be exercised in CI and on machines
without a Docker daemon.  Production deployments use :class:`DockerSandbox` (the default).
"""

from __future__ import annotations

import asyncio
import io
import os
import shutil
import signal
import tarfile
import time
import uuid
from pathlib import Path

from ..config import Settings
from .base import ExecResult, FileEntry, OutputCallback, Sandbox, SandboxStatus

MAX_CAPTURE = 2_000_000


class LocalSandbox(Sandbox):
    backend = "local"

    def __init__(self, session_id: str, settings: Settings) -> None:
        root = Path(settings.sandbox_local_root) / session_id.replace("-", "")[:12]
        super().__init__(session_id, str(root / "workspace"))
        self.s = settings
        self.root = root
        self._procs: dict[str, asyncio.subprocess.Process] = {}

    async def ensure(self) -> None:
        Path(self.workdir).mkdir(parents=True, exist_ok=True)

    async def stop(self) -> None:
        for p in list(self._procs.values()):
            self._terminate(p)

    async def destroy(self) -> None:
        await self.stop()
        shutil.rmtree(self.root, ignore_errors=True)

    async def status(self) -> SandboxStatus:
        state = "running" if Path(self.workdir).exists() else "absent"
        return SandboxStatus(self.backend, state, self.workdir, detail={"warning": "local backend is not isolated"})

    @staticmethod
    def _terminate(p: asyncio.subprocess.Process, sig: int = signal.SIGTERM) -> None:
        try:
            os.killpg(p.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: int = 120,
        env: dict[str, str] | None = None,
        on_output: OutputCallback | None = None,
        exec_id: str | None = None,
    ) -> ExecResult:
        await self.ensure()
        xid = exec_id or uuid.uuid4().hex[:10]
        timeout = max(1, min(int(timeout), self.s.sandbox_max_command_timeout))
        full_env = {**os.environ, "HOME": str(self.root), "LANG": "C.UTF-8", **(env or {})}
        started = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            "bash", "-c", command,
            cwd=cwd or self.workdir, env=full_env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        self._procs[xid] = proc
        out, err = bytearray(), bytearray()
        truncated = False

        async def pump(reader, buf: bytearray, stream: str) -> None:
            nonlocal truncated
            while True:
                chunk = await reader.read(4096)
                if not chunk:
                    return
                if len(buf) < MAX_CAPTURE:
                    buf.extend(chunk)
                else:
                    truncated = True
                if on_output is not None:
                    res = on_output(stream, chunk.decode("utf-8", "replace"))
                    if asyncio.iscoroutine(res):
                        await res

        timed_out = False
        tasks = [asyncio.create_task(pump(proc.stdout, out, "stdout")), asyncio.create_task(pump(proc.stderr, err, "stderr"))]
        try:
            await asyncio.wait_for(asyncio.gather(*tasks, proc.wait()), timeout=timeout)
        except asyncio.TimeoutError:
            timed_out = True
            self._terminate(proc, signal.SIGKILL)
            await proc.wait()
        except asyncio.CancelledError:
            self._terminate(proc, signal.SIGKILL)
            raise
        finally:
            self._procs.pop(xid, None)
            for t in tasks:
                t.cancel()
        return ExecResult(
            exit_code=proc.returncode if proc.returncode is not None else -1,
            stdout=out.decode("utf-8", "replace"), stderr=err.decode("utf-8", "replace"),
            timed_out=timed_out, truncated=truncated, duration=time.monotonic() - started,
        )

    async def kill(self, exec_id: str) -> None:
        p = self._procs.get(exec_id)
        if p:
            self._terminate(p, signal.SIGKILL)

    # ------------------------------------------------------------------ files
    async def read_bytes(self, path: str, max_bytes: int | None = None) -> bytes:
        p = Path(self.resolve(path))
        if p.is_dir():
            raise IsADirectoryError(str(p))
        with open(p, "rb") as f:
            return f.read(max_bytes) if max_bytes else f.read()

    async def write_bytes(self, path: str, data: bytes) -> None:
        p = Path(self.resolve(path))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    async def list_dir(self, path: str) -> list[FileEntry]:
        p = Path(self.resolve(path))
        entries = []
        if not p.is_dir():
            return entries
        for child in p.iterdir():
            try:
                st = child.stat()
            except OSError:
                continue
            entries.append(FileEntry(child.name, str(child), child.is_dir(), st.st_size, st.st_mtime))
        entries.sort(key=lambda e: (not e.is_dir, e.name.lower()))
        return entries

    async def exists(self, path: str) -> str | None:
        p = Path(self.resolve(path))
        return "dir" if p.is_dir() else "file" if p.exists() else None

    async def archive(self, path: str) -> bytes:
        p = Path(self.resolve(path))
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            tar.add(p, arcname=p.name)
        return buf.getvalue()

    def mcp_spec(self, argv: list[str], env: dict[str, str] | None = None) -> dict:
        return {"transport": "stdio", "command": argv[0], "args": argv[1:], "env": {**os.environ, **(env or {})}}
