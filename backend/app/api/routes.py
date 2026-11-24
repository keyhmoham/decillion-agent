from __future__ import annotations

import asyncio
import io
import mimetypes
import tarfile
import zipfile
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from .. import __version__
from ..agents.roles import ROLES
from ..llm.profile import DEFAULT_PROFILE
from ..llm.selftest import results_to_dicts, run_selftest
from ..mcp.catalog import PRESETS
from ..sandbox.base import SandboxError
from .deps import Container, get_container, require_auth, ws_authorized
from .schemas import (
    ApprovalBody, CoreMemoryBody, FileWrite, MemoryCreate, MemorySearch, SendMessage, SessionCreate, SessionPatch, TerminalExec,
)

router = APIRouter(prefix="/api")
protected = APIRouter(prefix="/api", dependencies=[Depends(require_auth)])

MAX_UPLOAD = 25 * 1024 * 1024
TEXT_PREVIEW_LIMIT = 1_500_000


async def _session_or_404(c: Container, sid: str) -> dict[str, Any]:
    try:
        s = await c.repo.get_session(sid)
    except ValueError:
        s = None
    if s is None:
        raise HTTPException(404, "Session not found")
    return s


# --------------------------------------------------------------------------- public
@router.get("/health")
async def health(c: Container = Depends(get_container)):
    model = await c.backend.health()
    db_ok = True
    try:
        async with c.db.session() as s:
            from sqlalchemy import text

            await s.execute(text("SELECT 1"))
    except Exception:  # noqa: BLE001
        db_ok = False
    return {"status": "ok" if db_ok and model.get("ok") else "degraded", "version": __version__, "database": db_ok, "model": model,
            "sandbox_backend": c.settings.sandbox_backend, "auth_required": bool(c.settings.app_api_key)}


# --------------------------------------------------------------------------- config / catalog
@protected.get("/config")
async def config(c: Container = Depends(get_container)):
    s = c.settings
    p = DEFAULT_PROFILE
    return {
        "version": __version__,
        "model": {"tag": s.llm_model, "family": p.architecture, "parameters": p.parameters, "quantization": p.quantization,
                  "hf_repo": p.hf_repo, "base_repo": p.base_repo, "backend": s.llm_backend, "num_ctx": s.llm_num_ctx,
                  "context_budget": s.context_budget, "thinking_budget": s.thinking_budget},
        "defaults": {"mode": s.default_mode, "thinking": s.enable_thinking, "max_steps": s.max_steps},
        "sandbox": {"backend": s.sandbox_backend, "image": s.sandbox_image, "memory": s.sandbox_memory, "cpus": s.sandbox_cpus, "network": s.sandbox_network},
        "mcp_catalog": [pr.to_dict(available=all(__import__("os").environ.get(r) for r in pr.requires)) for pr in PRESETS.values()],
        "roles": [{"name": r.name, "title": r.title, "description": r.description, "color": r.color} for r in ROLES.values()],
        "auth_required": bool(s.app_api_key),
    }


@protected.post("/model/selftest")
async def selftest(c: Container = Depends(get_container)):
    results = await run_selftest(c.service.make_model())
    return {"passed": all(r.passed for r in results), "results": results_to_dicts(results)}


# --------------------------------------------------------------------------- sessions
@protected.get("/sessions")
async def list_sessions(c: Container = Depends(get_container)):
    return await c.repo.list_sessions()


@protected.post("/sessions", status_code=201)
async def create_session(body: SessionCreate, c: Container = Depends(get_container)):
    s = await c.repo.create_session(body.title, body.config)
    return s


@protected.get("/sessions/{sid}")
async def get_session(sid: str, c: Container = Depends(get_container)):
    s = await _session_or_404(c, sid)
    return {**s, "running": c.service.is_running(sid)}


@protected.patch("/sessions/{sid}")
async def patch_session(sid: str, body: SessionPatch, c: Container = Depends(get_container)):
    await _session_or_404(c, sid)
    s = await c.repo.update_session(sid, title=body.title, config=body.config)
    return s  # MCP selection changes are applied lazily at the start of the next run


@protected.delete("/sessions/{sid}", status_code=204)
async def delete_session(sid: str, c: Container = Depends(get_container)):
    await _session_or_404(c, sid)
    await c.service.delete_session(sid)
    return Response(status_code=204)


@protected.get("/sessions/{sid}/messages")
async def messages(sid: str, c: Container = Depends(get_container)):
    await _session_or_404(c, sid)
    return await c.repo.list_messages(sid)


@protected.get("/sessions/{sid}/state")
async def state(sid: str, c: Container = Depends(get_container)):
    s = await _session_or_404(c, sid)
    sb = await c.sandboxes.get(sid).status()
    bus = c.service.bus(sid)
    return {
        "running": c.service.is_running(sid), "run_id": c.service.current_run_id(sid), "todos": s["todos"], "core_memory": s["core_memory"], "usage": s["usage"],
        "summary": s["summary"], "tasks": await c.repo.list_tasks(sid), "sandbox": sb.to_dict(), "mcp": c.mcp.status(sid), "last_seq": bus.seq,
    }


@protected.post("/sessions/{sid}/messages", status_code=202)
async def send_message(sid: str, body: SendMessage, c: Container = Depends(get_container)):
    await _session_or_404(c, sid)
    try:
        run_id = await c.service.start_run(sid, body.content, body.options or None)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"run_id": run_id}


@protected.post("/sessions/{sid}/cancel")
async def cancel(sid: str, c: Container = Depends(get_container)):
    return {"cancelled": await c.service.cancel(sid)}


@protected.post("/sessions/{sid}/approvals/{aid}")
async def approval(sid: str, aid: str, body: ApprovalBody, c: Container = Depends(get_container)):
    if not c.service.resolve_approval(sid, aid, body.approve, body.note):
        raise HTTPException(404, "No such pending approval")
    return {"ok": True}


# --------------------------------------------------------------------------- sandbox & files
@protected.get("/sessions/{sid}/sandbox")
async def sandbox_status(sid: str, c: Container = Depends(get_container)):
    await _session_or_404(c, sid)
    return (await c.sandboxes.get(sid).status()).to_dict()


@protected.post("/sessions/{sid}/sandbox/restart")
async def sandbox_restart(sid: str, c: Container = Depends(get_container)):
    await _session_or_404(c, sid)
    sb = c.sandboxes.get(sid)
    await sb.stop()
    await sb.ensure()
    return (await sb.status()).to_dict()


@protected.get("/sessions/{sid}/files")
async def list_files(sid: str, path: str = ".", c: Container = Depends(get_container)):
    await _session_or_404(c, sid)
    sb = c.sandboxes.get(sid)
    try:
        await sb.ensure()
        entries = await sb.list_dir(path)
    except SandboxError as exc:
        raise HTTPException(503, str(exc)) from exc
    return {"path": sb.resolve(path), "workdir": sb.workdir, "entries": [e.to_dict() for e in entries]}


@protected.get("/sessions/{sid}/file")
async def read_file(sid: str, path: str, raw: bool = False, c: Container = Depends(get_container)):
    await _session_or_404(c, sid)
    sb = c.sandboxes.get(sid)
    kind = await sb.exists(path)
    if kind is None:
        raise HTTPException(404, "No such file")
    if kind == "dir":
        raise HTTPException(400, "Path is a directory")
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    data = await sb.read_bytes(path, TEXT_PREVIEW_LIMIT if not raw else 25_000_000)
    if raw or mime.startswith(("image/", "application/pdf")):
        # Never let user-visible files execute script in our origin.
        return Response(data, media_type=mime, headers={"Content-Security-Policy": "sandbox; default-src 'none'; img-src data:; style-src 'unsafe-inline'", "X-Content-Type-Options": "nosniff"})
    binary = b"\x00" in data[:8192]
    return {"path": sb.resolve(path), "binary": binary, "size": len(data), "mime": mime, "content": "" if binary else data.decode("utf-8", "replace")}


@protected.put("/sessions/{sid}/file")
async def write_file(sid: str, body: FileWrite, c: Container = Depends(get_container)):
    await _session_or_404(c, sid)
    sb = c.sandboxes.get(sid)
    await sb.write_bytes(body.path, body.content.encode())
    return {"ok": True, "path": sb.resolve(body.path)}


@protected.post("/sessions/{sid}/upload")
async def upload(sid: str, dir: str = ".", files: list[UploadFile] = File(...), c: Container = Depends(get_container)):
    await _session_or_404(c, sid)
    sb = c.sandboxes.get(sid)
    saved = []
    for f in files:
        data = await f.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, f"{f.filename} exceeds {MAX_UPLOAD // 1024 // 1024} MB")
        name = (f.filename or "upload").replace("\\", "/").split("/")[-1]
        target = f"{sb.resolve(dir).rstrip('/')}/{name}"
        await sb.write_bytes(target, data)
        saved.append(target)
    return {"saved": saved}


@protected.get("/sessions/{sid}/download")
async def download(sid: str, path: str = ".", c: Container = Depends(get_container)):
    await _session_or_404(c, sid)
    sb = c.sandboxes.get(sid)
    tar_bytes = await sb.archive(path)
    out = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tar, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for m in tar.getmembers():
            if m.isfile():
                f = tar.extractfile(m)
                if f:
                    zf.writestr(m.name, f.read())
    out.seek(0)
    return StreamingResponse(out, media_type="application/zip", headers={"Content-Disposition": 'attachment; filename="workspace.zip"'})


@protected.post("/sessions/{sid}/terminal")
async def terminal(sid: str, body: TerminalExec, c: Container = Depends(get_container)):
    await _session_or_404(c, sid)
    sb = c.sandboxes.get(sid)
    res = await sb.exec(body.command, timeout=body.timeout, cwd=sb.resolve(body.cwd) if body.cwd else None)
    return {"exit_code": res.exit_code, "stdout": res.stdout, "stderr": res.stderr, "timed_out": res.timed_out, "duration": round(res.duration, 2)}


# --------------------------------------------------------------------------- memory
@protected.get("/memories")
async def list_memories(scope: str = "all", session_id: str | None = None, c: Container = Depends(get_container)):
    return await c.memory.list(session_id=session_id, scope=scope)


@protected.post("/memories", status_code=201)
async def create_memory(body: MemoryCreate, c: Container = Depends(get_container)):
    sid = body.session_id if body.scope == "session" else None
    return await c.memory.save(session_id=sid, kind=body.kind, content=body.content, tags=body.tags, importance=body.importance, source="user")


@protected.post("/memories/search")
async def search_memories(body: MemorySearch, c: Container = Depends(get_container)):
    return await c.memory.search(body.query, session_id=body.session_id, limit=body.limit, touch=False)


@protected.delete("/memories/{mid}", status_code=204)
async def delete_memory(mid: str, c: Container = Depends(get_container)):
    if not await c.memory.forget(mid):
        raise HTTPException(404, "No such memory")
    return Response(status_code=204)


@protected.put("/sessions/{sid}/core-memory/{block}")
async def set_core_memory(sid: str, block: str, body: CoreMemoryBody, c: Container = Depends(get_container)):
    if block not in ("user", "project", "notes"):
        raise HTTPException(400, "block must be user, project or notes")
    s = await _session_or_404(c, sid)
    core = {**s["core_memory"], block: body.content}
    await c.repo.set_fields(sid, core_memory=core)
    return core


@protected.get("/sessions/{sid}/tasks")
async def tasks(sid: str, c: Container = Depends(get_container)):
    await _session_or_404(c, sid)
    return await c.repo.list_tasks(sid)


# --------------------------------------------------------------------------- websocket
@router.websocket("/sessions/{sid}/ws")
async def ws_session(ws: WebSocket, sid: str, since: int = 0):
    c: Container = ws.app.state.container
    if not ws_authorized(ws, c.settings):
        await ws.close(code=4401)
        return
    try:
        session = await c.repo.get_session(sid)
    except ValueError:
        session = None
    if session is None:
        await ws.close(code=4404)
        return
    await ws.accept()
    bus = c.service.bus(sid)
    queue = bus.subscribe(since)
    await ws.send_json({"type": "hello", "running": c.service.is_running(sid), "last_seq": bus.seq, "session_id": sid})

    async def pump() -> None:
        while True:
            evt = await queue.get()
            await ws.send_json(evt)

    pump_task = asyncio.create_task(pump())
    try:
        while True:
            msg = await ws.receive_json()
            kind = msg.get("type")
            if kind == "send":
                try:
                    await c.service.start_run(sid, str(msg.get("content", "")).strip(), msg.get("options") or None)
                except RuntimeError as exc:
                    await ws.send_json({"type": "error", "message": str(exc)})
            elif kind == "cancel":
                await c.service.cancel(sid)
            elif kind == "approval":
                c.service.resolve_approval(sid, msg.get("approval_id", ""), bool(msg.get("approve")), msg.get("note", ""))
            elif kind == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        pass
    finally:
        pump_task.cancel()
        bus.unsubscribe(queue)


all_routers = [router, protected]
