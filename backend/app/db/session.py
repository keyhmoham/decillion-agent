"""Async engine / session factory and schema bootstrap."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from ..config import Settings
from .models import Base

log = logging.getLogger(__name__)

_ADVISORY_LOCK_KEY = 7_420_113  # serialises schema bootstrap between replicas


class Database:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.engine: AsyncEngine = create_async_engine(
            settings.database_url, pool_size=10, max_overflow=20, pool_pre_ping=True, pool_recycle=1800,
        )
        self.sessionmaker = async_sessionmaker(self.engine, expire_on_commit=False)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.sessionmaker() as s:
            yield s

    async def init(self) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": _ADVISORY_LOCK_KEY})
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
            await conn.run_sync(Base.metadata.create_all)
        log.info("database schema ready")

    @property
    def psycopg_url(self) -> str:
        """Plain libpq-style URL for langgraph's PostgresSaver (psycopg)."""
        url = self.settings.database_url
        return url.replace("postgresql+asyncpg://", "postgresql://").replace("postgresql+psycopg://", "postgresql://")

    async def dispose(self) -> None:
        await self.engine.dispose()
