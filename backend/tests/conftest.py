from __future__ import annotations

import os
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("SANDBOX_BACKEND", "local")

from app.llm.backends import GenerationRequest, StreamPiece  # noqa: E402
from app.llm.chat_model import A1ChatModel  # noqa: E402
from app.llm.profile import SamplingParams  # noqa: E402


class ScriptedBackend:
    """Backend that replays scripted generations, chunked like a real token stream."""

    name = "scripted"

    def __init__(self, scripts: list[str] | None = None, chunk: int = 7, fn=None) -> None:
        self.scripts = list(scripts or [])
        self.fn = fn
        self.chunk = chunk
        self.requests: list[GenerationRequest] = []

    async def stream(self, req: GenerationRequest) -> AsyncIterator[StreamPiece]:
        self.requests.append(req)
        text = self.fn(req, len(self.requests)) if self.fn else (self.scripts.pop(0) if self.scripts else "")
        for i in range(0, len(text), self.chunk):
            yield StreamPiece(text=text[i : i + self.chunk])
        yield StreamPiece(done=True, finish_reason="stop", prompt_tokens=len(req.prompt) // 4, completion_tokens=len(text) // 4)

    async def health(self):
        return {"ok": True}

    async def aclose(self):
        pass


@pytest.fixture
def make_model():
    def _make(scripts=None, fn=None, **kw) -> tuple[A1ChatModel, ScriptedBackend]:
        be = ScriptedBackend(scripts, fn=fn)
        model = A1ChatModel(backend=be, sampling=SamplingParams(), **kw)
        return model, be

    return _make


# ---------------------------------------------------------------- PostgreSQL fixtures
TEST_DB = os.environ.get("TEST_DATABASE_URL")
needs_pg = pytest.mark.skipif(not TEST_DB, reason="TEST_DATABASE_URL not set")


@pytest.fixture
async def pg_db():
    from sqlalchemy import text

    from app.config import Settings
    from app.db.models import Base
    from app.db.session import Database

    db = Database(Settings(database_url=TEST_DB))
    async with db.engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        await conn.run_sync(Base.metadata.drop_all)
    await db.init()
    yield db
    await db.dispose()
