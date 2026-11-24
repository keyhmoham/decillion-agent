"""Sandbox abstraction: the "computer" an agent session works on."""

from __future__ import annotations

import posixpath
import shlex
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

OutputCallback = Callable[[str, str], Awaitable[None] | None]  # (stream, text)


class SandboxError(RuntimeError):
    pass


@dataclass
class ExecResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    truncated: bool = False
    duration: float = 0.0

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


@dataclass
class FileEntry:
    name: str
    path: str
    is_dir: bool
    size: int = 0
    mtime: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "path": self.path, "is_dir": self.is_dir, "size": self.size, "mtime": self.mtime}


@dataclass
class SandboxStatus:
    backend: str
    state: str  # absent | running | stopped | error
    workdir: str
    container: str | None = None
    image: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend, "state": self.state, "workdir": self.workdir,
            "container": self.container, "image": self.image, **self.detail,
        }


class Sandbox(ABC):
    """One isolated computer per chat session."""

    backend: str = "abstract"

    def __init__(self, session_id: str, workdir: str) -> None:
        self.session_id = session_id
        self.workdir = workdir

    # -- lifecycle ----------------------------------------------------------------------
    @abstractmethod
    async def ensure(self) -> None:
        """Create/start the sandbox if necessary (idempotent)."""

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    async def destroy(self) -> None:
        """Remove the sandbox *and its workspace*."""

    @abstractmethod
    async def status(self) -> SandboxStatus: ...

    # -- execution ----------------------------------------------------------------------
    @abstractmethod
    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: int = 120,
        env: dict[str, str] | None = None,
        on_output: OutputCallback | None = None,
        exec_id: str | None = None,
    ) -> ExecResult: ...

    @abstractmethod
    async def kill(self, exec_id: str) -> None: ...

    # -- files --------------------------------------------------------------------------
    @abstractmethod
    async def read_bytes(self, path: str, max_bytes: int | None = None) -> bytes: ...

    @abstractmethod
    async def write_bytes(self, path: str, data: bytes) -> None: ...

    @abstractmethod
    async def list_dir(self, path: str) -> list[FileEntry]: ...

    @abstractmethod
    async def exists(self, path: str) -> str | None:
        """Return ``"file"``, ``"dir"`` or ``None``."""

    @abstractmethod
    async def archive(self, path: str) -> bytes:
        """tar.gz of a path (workspace download)."""

    # -- MCP ----------------------------------------------------------------------------
    @abstractmethod
    def mcp_stdio(self, argv: list[str], env: dict[str, str] | None = None) -> tuple[str, list[str], dict[str, str] | None]:
        """(command, args, env) that start ``argv`` *inside* this sandbox over stdio."""

    # -- helpers ------------------------------------------------------------------------
    def resolve(self, path: str) -> str:
        path = (path or ".").strip()
        if path.startswith("~"):
            path = path.replace("~", "/home/agent", 1)
        full = path if path.startswith("/") else posixpath.join(self.workdir, path)
        return posixpath.normpath(full)

    @staticmethod
    def quote(s: str) -> str:
        return shlex.quote(s)
