"""MCP over a generic byte pipe.

``mcp.client.stdio.stdio_client`` needs a local subprocess.  Our MCP servers live *inside* the session's
sandbox container, so instead of shelling out to the ``docker`` CLI (an extra binary to ship, architecture and
registry headaches) we talk to the Docker daemon through the Python SDK and adapt its exec stream to the
streams ``mcp.ClientSession`` expects.  Any object implementing :class:`StdioPipe` works (tests use a
socket pair that speaks Docker's multiplexed framing).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Protocol

import anyio
import mcp.types as types
from mcp.shared.message import SessionMessage

log = logging.getLogger(__name__)


class StdioPipe(Protocol):
    async def read(self) -> bytes | None:
        """Next chunk of the server's stdout, or ``None`` at EOF."""

    async def write(self, data: bytes) -> None: ...

    async def close(self) -> None: ...

    def stderr_tail(self) -> str: ...


@asynccontextmanager
async def pipe_stdio_client(pipe: StdioPipe) -> AsyncIterator[tuple]:
    """Yield ``(read_stream, write_stream)`` for ``ClientSession`` backed by ``pipe`` (newline-delimited JSON-RPC)."""
    read_w, read_r = anyio.create_memory_object_stream(0)
    write_w, write_r = anyio.create_memory_object_stream(0)

    async def reader() -> None:
        buf = b""
        try:
            async with read_w:
                while True:
                    chunk = await pipe.read()
                    if chunk is None:
                        return
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            msg = types.JSONRPCMessage.model_validate_json(line)
                        except Exception as exc:  # noqa: BLE001  (servers sometimes log to stdout)
                            log.debug("non-JSON-RPC line from MCP server: %r", line[:200])
                            await read_w.send(exc)
                            continue
                        await read_w.send(SessionMessage(msg))
        except anyio.ClosedResourceError:
            pass

    async def writer() -> None:
        try:
            async with write_r:
                async for session_message in write_r:
                    payload = session_message.message.model_dump_json(by_alias=True, exclude_none=True)
                    await pipe.write((payload + "\n").encode())
        except anyio.ClosedResourceError:
            pass

    async with anyio.create_task_group() as tg:
        tg.start_soon(reader)
        tg.start_soon(writer)
        try:
            yield read_r, write_w
        finally:
            await pipe.close()
            tg.cancel_scope.cancel()
