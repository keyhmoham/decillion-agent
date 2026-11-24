"""Owns the session -> sandbox mapping, idle reaping and start-up reconciliation."""

from __future__ import annotations

import asyncio
import logging
import time

from ..config import Settings
from .base import Sandbox
from .docker_sandbox import LABEL, DockerSandbox
from .local_sandbox import LocalSandbox

log = logging.getLogger(__name__)


class SandboxManager:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self._boxes: dict[str, Sandbox] = {}
        self._last_used: dict[str, float] = {}
        self._reaper: asyncio.Task | None = None
        self._docker_client = None

    # ------------------------------------------------------------------ factory
    def _make(self, session_id: str) -> Sandbox:
        if self.s.sandbox_backend == "local":
            return LocalSandbox(session_id, self.s)
        return DockerSandbox(session_id, self.s, client=self._docker_client)

    def get(self, session_id: str) -> Sandbox:
        """Return the (lazily created) sandbox object. Call ``await sb.ensure()`` to start it."""
        box = self._boxes.get(session_id)
        if box is None:
            box = self._boxes[session_id] = self._make(session_id)
        self._last_used[session_id] = time.monotonic()
        return box

    def touch(self, session_id: str) -> None:
        self._last_used[session_id] = time.monotonic()

    async def destroy(self, session_id: str) -> None:
        box = self._boxes.pop(session_id, None) or self._make(session_id)
        self._last_used.pop(session_id, None)
        try:
            await box.destroy()
        except Exception:  # noqa: BLE001
            log.exception("failed to destroy sandbox for %s", session_id)

    # ------------------------------------------------------------------ background reaper
    async def start(self) -> None:
        if self._reaper is None:
            self._reaper = asyncio.create_task(self._reap_loop(), name="sandbox-reaper")

    async def shutdown(self) -> None:
        if self._reaper:
            self._reaper.cancel()
            self._reaper = None

    async def _reap_loop(self) -> None:
        idle = self.s.sandbox_idle_minutes * 60
        while True:
            await asyncio.sleep(60)
            now = time.monotonic()
            for sid, box in list(self._boxes.items()):
                if now - self._last_used.get(sid, now) > idle:
                    try:
                        st = await box.status()
                        if st.state == "running":
                            log.info("stopping idle sandbox %s", sid)
                            await box.stop()
                    except Exception:  # noqa: BLE001
                        log.exception("reaper failed for %s", sid)

    async def list_orphans(self, known_session_ids: set[str]) -> list[str]:
        """Docker containers labelled as ours whose session no longer exists."""
        if self.s.sandbox_backend != "docker":
            return []

        def _scan() -> list[str]:
            import docker

            client = docker.from_env()
            found = []
            for c in client.containers.list(all=True, filters={"label": LABEL}):
                sid = (c.labels or {}).get("decillion.session")
                if sid and sid not in known_session_ids:
                    found.append(sid)
            return found

        try:
            return await asyncio.to_thread(_scan)
        except Exception:  # noqa: BLE001
            log.warning("could not scan for orphaned sandboxes", exc_info=True)
            return []
