"""Long-term memory store.

Retrieval needs no embedding model (an extra model would not fit the 4B-on-a-laptop story):
Postgres full-text search (``ts_rank_cd``) is blended with ``pg_trgm`` similarity, an importance
weight and a recency decay.  Writes de-duplicate near-identical memories instead of piling up.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import delete, func, select, text, update

from ..db.models import Memory
from ..db.session import Database

KINDS = ("fact", "preference", "episode", "lesson", "project")

_STOP = set(
    "the a an and or of to in on for with is are was were be been it this that these those i you he she we they my your our "
    "can could should would will do does did have has had not no yes please just about into from as at by what which who how why when where".split()
)


def to_or_tsquery(query: str) -> str | None:
    words = [w for w in re.findall(r"[A-Za-z0-9_]{3,}", query.lower()) if w not in _STOP]
    seen: list[str] = []
    for w in words:
        if w not in seen:
            seen.append(w)
    return " | ".join(seen[:16]) or None


def _to_dict(m: Memory, score: float | None = None) -> dict[str, Any]:
    d = {
        "id": str(m.id), "session_id": str(m.session_id) if m.session_id else None, "kind": m.kind,
        "content": m.content, "tags": list(m.tags or []), "importance": m.importance, "source": m.source,
        "access_count": m.access_count, "created_at": m.created_at.isoformat() if m.created_at else None,
        "updated_at": m.updated_at.isoformat() if m.updated_at else None,
    }
    if score is not None:
        d["score"] = round(float(score), 4)
    return d


class MemoryStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def save(
        self, *, session_id: str | None, kind: str = "fact", content: str, tags: list[str] | None = None,
        importance: int = 3, source: str = "agent",
    ) -> dict[str, Any]:
        content = content.strip()
        if not content:
            raise ValueError("memory content is empty")
        kind = kind if kind in KINDS else "fact"
        importance = max(1, min(5, int(importance)))
        sid = uuid.UUID(session_id) if session_id else None
        async with self.db.session() as s:
            # near-duplicate? -> merge instead of insert
            dup = (await s.execute(
                text(
                    "SELECT id FROM memories WHERE (session_id IS NOT DISTINCT FROM :sid) AND kind = :kind "
                    "AND similarity(content, :c) > 0.72 ORDER BY similarity(content, :c) DESC LIMIT 1"
                ),
                {"sid": sid, "kind": kind, "c": content},
            )).first()
            if dup:
                m = await s.get(Memory, dup[0])
                assert m is not None
                m.content = content
                m.importance = max(m.importance, importance)
                m.tags = sorted(set((m.tags or []) + (tags or [])))
                m.updated_at = datetime.now(timezone.utc)
                await s.commit()
                return {**_to_dict(m), "deduplicated": True}
            m = Memory(session_id=sid, kind=kind, content=content, tags=sorted(set(tags or [])), importance=importance, source=source)
            s.add(m)
            await s.commit()
            await s.refresh(m)
            return _to_dict(m)

    async def search(
        self, query: str, *, session_id: str | None = None, kind: str | None = None, limit: int = 5,
        include_global: bool = True, touch: bool = True,
    ) -> list[dict[str, Any]]:
        tsq = to_or_tsquery(query)
        sid = uuid.UUID(session_id) if session_id else None
        scope = "(m.session_id IS NULL OR m.session_id = :sid)" if (sid and include_global) else (
            "m.session_id = :sid" if sid else "m.session_id IS NULL")
        kind_sql = "AND m.kind = :kind" if kind else ""
        sql = text(
            f"""
            SELECT m.id, (
                COALESCE(CASE WHEN CAST(:tsq AS text) IS NULL THEN 0 ELSE ts_rank_cd(m.tsv, to_tsquery('english', CAST(:tsq AS text)), 32) END, 0) * 2.0
                + similarity(m.content, :q) * 1.0
                + m.importance * 0.08
                + 0.25 * exp(-extract(epoch FROM (now() - m.updated_at)) / 2592000.0)
            ) AS score
            FROM memories m
            WHERE {scope} {kind_sql}
              AND ( (CAST(:tsq AS text) IS NOT NULL AND m.tsv @@ to_tsquery('english', CAST(:tsq AS text))) OR similarity(m.content, :q) > 0.12 )
            ORDER BY score DESC LIMIT :limit
            """
        )
        params: dict[str, Any] = {"tsq": tsq, "q": query, "sid": sid, "limit": limit}
        if kind:
            params["kind"] = kind
        async with self.db.session() as s:
            rows = (await s.execute(sql, params)).all()
            if not rows:
                return []
            ids = [r[0] for r in rows]
            scores = {r[0]: r[1] for r in rows}
            mems = {m.id: m for m in (await s.execute(select(Memory).where(Memory.id.in_(ids)))).scalars()}
            if touch:
                await s.execute(update(Memory).where(Memory.id.in_(ids)).values(
                    access_count=Memory.access_count + 1, last_accessed_at=datetime.now(timezone.utc)))
                await s.commit()
            return [_to_dict(mems[i], scores[i]) for i in ids if i in mems]

    async def list(self, *, session_id: str | None = None, scope: str = "all", limit: int = 200) -> list[dict[str, Any]]:
        async with self.db.session() as s:
            q = select(Memory).order_by(Memory.updated_at.desc()).limit(limit)
            sid = uuid.UUID(session_id) if session_id else None
            if scope == "global":
                q = q.where(Memory.session_id.is_(None))
            elif scope == "session" and sid:
                q = q.where(Memory.session_id == sid)
            elif sid:
                q = q.where((Memory.session_id.is_(None)) | (Memory.session_id == sid))
            return [_to_dict(m) for m in (await s.execute(q)).scalars()]

    async def resolve_prefix(self, prefix: str) -> str | None:
        """Expand the 8-character id prefix shown to the model into the full UUID."""
        if not re.fullmatch(r"[0-9a-fA-F-]{4,36}", prefix):
            return None
        async with self.db.session() as s:
            row = (await s.execute(text("SELECT id::text FROM memories WHERE id::text LIKE :p LIMIT 2"), {"p": prefix.lower() + "%"})).all()
            return row[0][0] if len(row) == 1 else None

    async def forget(self, memory_id: str) -> bool:
        try:
            mid = uuid.UUID(memory_id)
        except ValueError:
            return False
        async with self.db.session() as s:
            res = await s.execute(delete(Memory).where(Memory.id == mid))
            await s.commit()
            return (res.rowcount or 0) > 0

    async def consolidate(self, *, max_age_days: int = 90) -> dict[str, int]:
        """Housekeeping: drop never-used, low-importance, stale episodic memories."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        async with self.db.session() as s:
            res = await s.execute(
                delete(Memory).where(Memory.importance <= 2, Memory.access_count == 0, Memory.updated_at < cutoff, Memory.kind == "episode")
            )
            await s.commit()
            total = (await s.execute(select(func.count()).select_from(Memory))).scalar_one()
        return {"deleted": res.rowcount or 0, "remaining": int(total)}
