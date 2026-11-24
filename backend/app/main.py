"""FastAPI application and composition root."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .agents.service import AgentService
from .api.deps import Container
from .api.routes import all_routers
from .config import Settings, get_settings
from .db.repo import Repo
from .db.session import Database
from .llm.backends import make_backend
from .mcp.manager import McpManager
from .memory.store import MemoryStore
from .sandbox.manager import SandboxManager

log = logging.getLogger("decillion")


async def build_container(settings: Settings, db: Database, *, checkpointer=None) -> Container:
    c = Container()
    c.settings = settings
    c.db = db
    c.repo = Repo(c.db)
    c.memory = MemoryStore(c.db)
    c.sandboxes = SandboxManager(settings)
    c.mcp = McpManager(settings)
    c.backend = make_backend(settings)
    c.service = AgentService(settings, c.repo, c.memory, c.sandboxes, c.mcp, c.backend, checkpointer)
    n = await c.repo.fail_dangling_runs()
    if n:
        log.warning("marked %d interrupted run(s) as failed", n)
    await c.sandboxes.start()
    return c


@asynccontextmanager
async def _checkpointer(settings: Settings, db: Database):
    """LangGraph Postgres checkpointer (conversation state per session)."""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    pool = AsyncConnectionPool(
        db.psycopg_url, min_size=1, max_size=8, open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    await pool.open()
    try:
        saver = AsyncPostgresSaver(pool)
        await saver.setup()
        yield saver
    finally:
        await pool.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db = Database(settings)
        await db.init()
        async with _checkpointer(settings, db) as saver:
            container = await build_container(settings, db, checkpointer=saver)
            app.state.container = container
            log.info("Decillion Agent %s ready (model=%s, backend=%s, sandbox=%s)", __version__, settings.llm_model, settings.llm_backend, settings.sandbox_backend)
            try:
                yield
            finally:
                await container.service.shutdown()
                await container.sandboxes.shutdown()
                await container.backend.aclose()
        await db.dispose()

    app = FastAPI(title=settings.app_name, version=__version__, lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_list, allow_methods=["*"], allow_headers=["*"], allow_credentials=False)
    for r in all_routers:
        app.include_router(r)
    return app


app = create_app() if __name__ != "__main__" else None  # uvicorn app.main:app
