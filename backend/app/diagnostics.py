"""Operator diagnostics: answers "why does X not work?" with concrete, checkable facts."""

from __future__ import annotations

import asyncio
from typing import Any

from .config import Settings
from .mcp.catalog import PRESETS
from .sandbox.base import Sandbox


def _row(name: str, ok: bool | None, detail: str, fix: str = "") -> dict[str, Any]:
    return {"name": name, "ok": ok, "detail": detail, "fix": fix}


async def run_diagnostics(settings: Settings, *, search: Any = None, sandbox: Sandbox | None = None, probe_search: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # ---- sandbox backend --------------------------------------------------------------------------
    if settings.sandbox_backend == "docker":
        def _daemon() -> tuple[bool, str]:
            try:
                import docker

                c = docker.from_env(timeout=5)
                c.ping()
                try:
                    c.images.get(settings.sandbox_image)
                    return True, f"daemon reachable; image {settings.sandbox_image} present"
                except Exception:  # noqa: BLE001
                    return False, f"daemon reachable but image {settings.sandbox_image} is missing"
            except Exception as exc:  # noqa: BLE001
                return False, f"cannot reach the Docker daemon: {exc}"

        ok, detail = await asyncio.to_thread(_daemon)
        rows.append(_row("docker daemon + sandbox image", ok, detail,
                         "" if ok else "Mount /var/run/docker.sock into the backend and run: docker compose build sandbox-image"))
    else:
        rows.append(_row("sandbox backend", None, "local backend (development only, not isolated)"))

    # ---- MCP binaries inside this session's sandbox -------------------------------------------------
    if sandbox is not None:
        missing: list[str] = []
        for p in PRESETS.values():
            res = await sandbox.exec(f"command -v {p.binary}", timeout=15)
            if not res.ok:
                missing.append(p.binary)
        rows.append(_row("MCP servers in sandbox image", not missing,
                         "all preset servers installed" if not missing else "missing: " + ", ".join(missing),
                         "" if not missing else "Rebuild the image (docker compose build sandbox-image) and start a new session."))

    # ---- web search ---------------------------------------------------------------------------------
    if search is not None:
        names = [p.name for p in search.providers]
        rows.append(_row("web search providers", bool(names), " → ".join(names),
                         "" if settings.searxng_url or settings.brave_api_key else
                         "Only scraping providers are configured; set SEARXNG_URL (bundled in docker-compose) or BRAVE_API_KEY for reliable search."))
        if probe_search:
            try:
                results, who = await search.search("python programming language", 3)
                rows.append(_row("web search probe", True, f"{len(results)} results via {who}"))
            except Exception as exc:  # noqa: BLE001
                rows.append(_row("web search probe", False, str(exc)[:400]))
    return rows
