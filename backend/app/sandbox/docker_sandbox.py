"""Docker sandbox: one hardened container + one workspace volume per chat session."""

from __future__ import annotations

import asyncio
import io
import logging
import tarfile
import time
import uuid
from typing import Any

from ..config import Settings
from .base import ExecResult, FileEntry, OutputCallback, Sandbox, SandboxError, SandboxStatus

log = logging.getLogger(__name__)

MAX_CAPTURE = 2_000_000  # bytes kept per stream
LABEL = "decillion.sandbox"


def _import_docker():
    try:
        import docker  # type: ignore
        import docker.errors  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise SandboxError("The 'docker' python package is required for the docker sandbox") from exc
    return docker


class DockerSandbox(Sandbox):
    backend = "docker"

    def __init__(self, session_id: str, settings: Settings, client: Any | None = None) -> None:
        super().__init__(session_id, settings.sandbox_workdir)
        self.s = settings
        self._client = client
        short = session_id.replace("-", "")[:12]
        self.name = f"a1-sbx-{short}"
        self.volume = f"a1-ws-{short}"
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ client / lifecycle
    @property
    def client(self):
        if self._client is None:
            docker = _import_docker()
            self._client = docker.from_env(timeout=settings_timeout(self.s))
        return self._client

    def _get(self):
        docker = _import_docker()
        try:
            return self.client.containers.get(self.name)
        except docker.errors.NotFound:
            return None

    def _create_sync(self):
        docker = _import_docker()
        c = self._get()
        if c is None:
            log.info("creating sandbox container %s from %s", self.name, self.s.sandbox_image)
            try:
                c = self.client.containers.run(
                    self.s.sandbox_image,
                    command=["sleep", "infinity"],
                    name=self.name,
                    detach=True,
                    init=True,
                    user=self.s.sandbox_user,
                    working_dir=self.workdir,
                    hostname="sandbox",
                    mem_limit=self.s.sandbox_memory,
                    memswap_limit=self.s.sandbox_memory,
                    nano_cpus=int(self.s.sandbox_cpus * 1e9),
                    pids_limit=self.s.sandbox_pids_limit,
                    cap_drop=["ALL"],
                    security_opt=["no-new-privileges:true"],
                    volumes={self.volume: {"bind": self.workdir, "mode": "rw"}},
                    network_mode=self.s.sandbox_network,
                    labels={LABEL: "1", "decillion.session": self.session_id},
                    environment={"HOME": "/home/agent", "LANG": "C.UTF-8", "A1_SESSION": self.session_id},
                    restart_policy={"Name": "no"},
                )
            except docker.errors.ImageNotFound as exc:
                raise SandboxError(
                    f"Sandbox image '{self.s.sandbox_image}' not found. Build it with "
                    f"`docker compose build sandbox-image`."
                ) from exc
            except docker.errors.APIError as exc:
                raise SandboxError(f"Docker refused to create the sandbox: {exc.explanation or exc}") from exc
        else:
            c.reload()
            if c.status != "running":
                c.start()
        return c

    async def ensure(self) -> None:
        async with self._lock:
            c = await asyncio.to_thread(self._get)
            if c is not None:
                await asyncio.to_thread(c.reload)
                if c.status == "running":
                    return
            await asyncio.to_thread(self._create_sync)

    async def stop(self) -> None:
        c = await asyncio.to_thread(self._get)
        if c is not None:
            await asyncio.to_thread(c.stop, timeout=3)

    async def destroy(self) -> None:
        docker = _import_docker()

        def _do():
            c = self._get()
            if c is not None:
                c.remove(force=True, v=False)
            try:
                self.client.volumes.get(self.volume).remove(force=True)
            except docker.errors.NotFound:
                pass
            except docker.errors.APIError as exc:
                log.warning("could not remove volume %s: %s", self.volume, exc)

        await asyncio.to_thread(_do)

    async def status(self) -> SandboxStatus:
        try:
            c = await asyncio.to_thread(self._get)
        except Exception as exc:  # noqa: BLE001
            return SandboxStatus(self.backend, "error", self.workdir, detail={"error": str(exc)})
        if c is None:
            return SandboxStatus(self.backend, "absent", self.workdir, image=self.s.sandbox_image)
        await asyncio.to_thread(c.reload)
        return SandboxStatus(
            self.backend, "running" if c.status == "running" else "stopped", self.workdir,
            container=c.short_id, image=self.s.sandbox_image,
            detail={"memory": self.s.sandbox_memory, "cpus": self.s.sandbox_cpus, "network": self.s.sandbox_network},
        )

    # ------------------------------------------------------------------ exec
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
        environment = {**(env or {}), "A1_CMD": command}
        # `timeout` guarantees the command ends; `setsid` lets us kill the whole process tree.
        # The command travels in the environment (A1_CMD) so no quoting layer can corrupt it; the
        # `a1x-<id>` marker (bash's $0) lets `kill()` find the process tree again.
        argv = ["setsid", "-w", "timeout", "-k", "3", str(timeout), "bash", "-c", 'eval "$A1_CMD"', f"a1x-{xid}"]
        loop = asyncio.get_running_loop()
        started = time.monotonic()

        def emit(stream: str, text: str) -> None:
            if on_output is None:
                return

            def _call():
                res = on_output(stream, text)
                if asyncio.iscoroutine(res):
                    asyncio.ensure_future(res)

            loop.call_soon_threadsafe(_call)

        def run() -> ExecResult:
            api = self.client.api
            c = self._get()
            if c is None:
                raise SandboxError("sandbox container disappeared")
            ex = api.exec_create(
                c.id, argv, environment=environment, workdir=cwd or self.workdir,
                user=self.s.sandbox_user, stdout=True, stderr=True, tty=False,
            )
            out, err = bytearray(), bytearray()
            truncated = False
            for so, se in api.exec_start(ex["Id"], stream=True, demux=True):
                if so:
                    if len(out) < MAX_CAPTURE:
                        out.extend(so)
                    else:
                        truncated = True
                    emit("stdout", so.decode("utf-8", "replace"))
                if se:
                    if len(err) < MAX_CAPTURE:
                        err.extend(se)
                    else:
                        truncated = True
                    emit("stderr", se.decode("utf-8", "replace"))
            info = api.exec_inspect(ex["Id"])
            code = info.get("ExitCode")
            code = -1 if code is None else code
            return ExecResult(
                exit_code=code,
                stdout=out.decode("utf-8", "replace"),
                stderr=err.decode("utf-8", "replace"),
                timed_out=code in (124, 137) and (time.monotonic() - started) >= timeout - 0.5,
                truncated=truncated,
                duration=time.monotonic() - started,
            )

        try:
            return await asyncio.to_thread(run)
        except asyncio.CancelledError:
            await self.kill(xid)
            raise

    async def kill(self, exec_id: str) -> None:
        def _kill():
            c = self._get()
            if c is None:
                return
            c.exec_run(["pkill", "-TERM", "-f", f"a1x-{exec_id}"], user=self.s.sandbox_user)
            time.sleep(1.5)
            c.exec_run(["pkill", "-KILL", "-f", f"a1x-{exec_id}"], user=self.s.sandbox_user)

        await asyncio.to_thread(_kill)

    # ------------------------------------------------------------------ files
    async def read_bytes(self, path: str, max_bytes: int | None = None) -> bytes:
        await self.ensure()
        full = self.resolve(path)

        def _read() -> bytes:
            docker = _import_docker()
            c = self._get()
            try:
                stream, _stat = c.get_archive(full)
            except docker.errors.NotFound as exc:
                raise FileNotFoundError(full) from exc
            buf = io.BytesIO(b"".join(stream))
            with tarfile.open(fileobj=buf) as tar:
                member = next((m for m in tar.getmembers() if m.isfile()), None)
                if member is None:
                    raise IsADirectoryError(full)
                f = tar.extractfile(member)
                assert f is not None
                return f.read(max_bytes) if max_bytes else f.read()

        return await asyncio.to_thread(_read)

    async def write_bytes(self, path: str, data: bytes) -> None:
        await self.ensure()
        full = self.resolve(path)
        parent, _, name = full.rpartition("/")
        res = await self.exec(f"mkdir -p {self.quote(parent or '/')}", timeout=15)
        if not res.ok:
            raise SandboxError(f"cannot create directory {parent}: {res.stderr.strip()}")

        def _write():
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as tar:
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                info.mtime = int(time.time())
                info.mode = 0o644
                info.uid = info.gid = 1000
                tar.addfile(info, io.BytesIO(data))
            buf.seek(0)
            c = self._get()
            if not c.put_archive(parent or "/", buf.getvalue()):
                raise SandboxError(f"failed to write {full}")

        await asyncio.to_thread(_write)

    async def list_dir(self, path: str) -> list[FileEntry]:
        full = self.resolve(path)
        res = await self.exec(
            f"find {self.quote(full)} -mindepth 1 -maxdepth 1 -printf '%y\\t%s\\t%T@\\t%f\\n' 2>/dev/null | sort -t$'\\t' -k4",
            timeout=20,
        )
        entries: list[FileEntry] = []
        for line in res.stdout.splitlines():
            parts = line.split("\t", 3)
            if len(parts) != 4:
                continue
            kind, size, mtime, name = parts
            entries.append(FileEntry(name, f"{full.rstrip('/')}/{name}", kind == "d", int(size or 0), float(mtime or 0)))
        entries.sort(key=lambda e: (not e.is_dir, e.name.lower()))
        return entries

    async def exists(self, path: str) -> str | None:
        full = self.quote(self.resolve(path))
        res = await self.exec(f"if [ -d {full} ]; then echo dir; elif [ -e {full} ]; then echo file; fi", timeout=10)
        out = res.stdout.strip()
        return out or None

    async def archive(self, path: str) -> bytes:
        await self.ensure()
        full = self.resolve(path)

        def _arch() -> bytes:
            c = self._get()
            stream, _ = c.get_archive(full)
            return b"".join(stream)

        return await asyncio.to_thread(_arch)

    # ------------------------------------------------------------------ MCP
    def mcp_stdio(self, argv: list[str], env: dict[str, str] | None = None) -> tuple[str, list[str], dict[str, str] | None]:
        args = ["exec", "-i", "-w", self.workdir, "-u", self.s.sandbox_user]
        for k, v in (env or {}).items():
            args += ["-e", f"{k}={v}"]
        args += [self.name, *argv]
        return "docker", args, None


def settings_timeout(s: Settings) -> int:
    return s.sandbox_max_command_timeout + 60
