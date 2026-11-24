"""Repository helpers over the SQLAlchemy models (all async)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, func, select

from .models import ChatSession, Message, Run, TeamTask
from .session import Database

DEFAULT_CONFIG: dict[str, Any] = {
    "mode": "auto",            # auto | agent | team
    "thinking": True,
    "approval": "auto",        # auto | ask
    "mcp": ["time", "git", "sequential-thinking"],
    "custom_mcp": [],
    "team_parallelism": 2,
    "memory": True,
}


def _uuid(v: str | uuid.UUID) -> uuid.UUID:
    return v if isinstance(v, uuid.UUID) else uuid.UUID(str(v))


def session_dict(s: ChatSession) -> dict[str, Any]:
    return {
        "id": str(s.id), "title": s.title, "created_at": s.created_at.isoformat(), "updated_at": s.updated_at.isoformat(),
        "config": {**DEFAULT_CONFIG, **(s.config or {})}, "todos": s.todos or [], "core_memory": s.core_memory or {},
        "summary": s.summary or "", "usage": s.usage or {}, "archived": s.archived,
    }


def message_dict(m: Message) -> dict[str, Any]:
    return {
        "id": m.id, "session_id": str(m.session_id), "run_id": str(m.run_id) if m.run_id else None, "role": m.role,
        "agent": m.agent, "content": m.content, "reasoning": m.reasoning, "tool_calls": m.tool_calls or [],
        "tool_call_id": m.tool_call_id, "tool_name": m.tool_name, "meta": m.meta or {}, "created_at": m.created_at.isoformat(),
    }


def task_dict(t: TeamTask) -> dict[str, Any]:
    return {
        "id": str(t.id), "key": t.key, "title": t.title, "description": t.description, "role": t.role, "status": t.status,
        "depends_on": t.depends_on or [], "result": t.result, "attempts": t.attempts, "handoff_from": t.handoff_from,
        "run_id": str(t.run_id) if t.run_id else None,
        "started_at": t.started_at.isoformat() if t.started_at else None,
        "finished_at": t.finished_at.isoformat() if t.finished_at else None,
    }


class Repo:
    def __init__(self, db: Database) -> None:
        self.db = db

    # ---------------------------------------------------------------- sessions
    async def create_session(self, title: str = "New session", config: dict[str, Any] | None = None) -> dict[str, Any]:
        async with self.db.session() as s:
            obj = ChatSession(title=title, config={**DEFAULT_CONFIG, **(config or {})})
            s.add(obj)
            await s.commit()
            await s.refresh(obj)
            return session_dict(obj)

    async def list_sessions(self, limit: int = 100) -> list[dict[str, Any]]:
        async with self.db.session() as s:
            rows = (await s.execute(
                select(ChatSession).where(ChatSession.archived.is_(False)).order_by(ChatSession.updated_at.desc()).limit(limit)
            )).scalars().all()
            return [session_dict(r) for r in rows]

    async def get_session(self, sid: str | uuid.UUID) -> dict[str, Any] | None:
        async with self.db.session() as s:
            obj = await s.get(ChatSession, _uuid(sid))
            return session_dict(obj) if obj else None

    async def update_session(self, sid: str, *, title: str | None = None, config: dict[str, Any] | None = None) -> dict[str, Any] | None:
        async with self.db.session() as s:
            obj = await s.get(ChatSession, _uuid(sid))
            if obj is None:
                return None
            if title is not None:
                obj.title = title[:200]
            if config is not None:
                obj.config = {**DEFAULT_CONFIG, **(obj.config or {}), **config}
            obj.updated_at = datetime.now(timezone.utc)
            await s.commit()
            return session_dict(obj)

    async def set_fields(self, sid: str, **fields: Any) -> None:
        async with self.db.session() as s:
            obj = await s.get(ChatSession, _uuid(sid))
            if obj is None:
                return
            for k, v in fields.items():
                setattr(obj, k, v)
            obj.updated_at = datetime.now(timezone.utc)
            await s.commit()

    async def delete_session(self, sid: str) -> bool:
        async with self.db.session() as s:
            res = await s.execute(delete(ChatSession).where(ChatSession.id == _uuid(sid)))
            await s.commit()
            return (res.rowcount or 0) > 0

    async def all_session_ids(self) -> set[str]:
        async with self.db.session() as s:
            return {str(r) for r in (await s.execute(select(ChatSession.id))).scalars()}

    # ---------------------------------------------------------------- messages
    async def add_message(self, sid: str, role: str, **fields: Any) -> dict[str, Any]:
        async with self.db.session() as s:
            m = Message(session_id=_uuid(sid), role=role, **fields)
            s.add(m)
            await s.commit()
            await s.refresh(m)
            return message_dict(m)

    async def list_messages(self, sid: str, limit: int = 2000) -> list[dict[str, Any]]:
        async with self.db.session() as s:
            rows = (await s.execute(
                select(Message).where(Message.session_id == _uuid(sid)).order_by(Message.id).limit(limit)
            )).scalars().all()
            return [message_dict(r) for r in rows]

    async def count_messages(self, sid: str) -> int:
        async with self.db.session() as s:
            return int((await s.execute(select(func.count()).select_from(Message).where(Message.session_id == _uuid(sid)))).scalar_one())

    # ---------------------------------------------------------------- runs
    async def create_run(self, sid: str, mode: str, goal: str) -> str:
        async with self.db.session() as s:
            r = Run(session_id=_uuid(sid), mode=mode, goal=goal[:2000])
            s.add(r)
            await s.commit()
            return str(r.id)

    async def finish_run(self, run_id: str, status: str, error: str | None = None, usage: dict[str, Any] | None = None) -> None:
        async with self.db.session() as s:
            r = await s.get(Run, _uuid(run_id))
            if r:
                r.status, r.error, r.finished_at = status, error, datetime.now(timezone.utc)
                r.usage = usage or {}
                await s.commit()

    async def fail_dangling_runs(self) -> int:
        """On start-up: runs left 'running' by a crashed process are marked failed."""
        async with self.db.session() as s:
            rows = (await s.execute(select(Run).where(Run.status == "running"))).scalars().all()
            for r in rows:
                r.status, r.error, r.finished_at = "failed", "backend restarted while the run was active", datetime.now(timezone.utc)
            await s.commit()
            return len(rows)

    # ---------------------------------------------------------------- tasks
    async def upsert_task(self, sid: str, run_id: str | None, key: str, **fields: Any) -> dict[str, Any]:
        async with self.db.session() as s:
            q = select(TeamTask).where(TeamTask.session_id == _uuid(sid), TeamTask.key == key)
            if run_id:
                q = q.where(TeamTask.run_id == _uuid(run_id))
            t = (await s.execute(q)).scalars().first()
            if t is None:
                t = TeamTask(session_id=_uuid(sid), run_id=_uuid(run_id) if run_id else None, key=key)
                s.add(t)
            for k, v in fields.items():
                setattr(t, k, v)
            await s.commit()
            await s.refresh(t)
            return task_dict(t)

    async def list_tasks(self, sid: str, run_id: str | None = None) -> list[dict[str, Any]]:
        async with self.db.session() as s:
            q = select(TeamTask).where(TeamTask.session_id == _uuid(sid)).order_by(TeamTask.created_at, TeamTask.key)
            if run_id:
                q = q.where(TeamTask.run_id == _uuid(run_id))
            return [task_dict(t) for t in (await s.execute(q)).scalars()]


class DbSessionStore:
    """`SessionStore` implementation used by tools: todos and core-memory blocks live on the session row."""

    def __init__(self, repo: Repo, session_id: str) -> None:
        self.repo, self.session_id = repo, session_id

    async def get_todos(self) -> list[dict[str, Any]]:
        s = await self.repo.get_session(self.session_id)
        return (s or {}).get("todos", [])

    async def set_todos(self, todos: list[dict[str, Any]]) -> None:
        await self.repo.set_fields(self.session_id, todos=todos)

    async def get_core_memory(self) -> dict[str, str]:
        s = await self.repo.get_session(self.session_id)
        return (s or {}).get("core_memory", {})

    async def set_core_block(self, block: str, text: str) -> None:
        cm = await self.get_core_memory()
        cm[block] = text
        await self.repo.set_fields(self.session_id, core_memory=cm)


class InMemorySessionStore:
    """Used in unit tests."""

    def __init__(self) -> None:
        self.todos: list[dict[str, Any]] = []
        self.core: dict[str, str] = {}

    async def get_todos(self):
        return self.todos

    async def set_todos(self, todos):
        self.todos = todos

    async def get_core_memory(self):
        return self.core

    async def set_core_block(self, block, text):
        self.core[block] = text
