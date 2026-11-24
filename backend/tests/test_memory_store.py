import pytest

from app.db.repo import Repo
from app.memory.store import MemoryStore, to_or_tsquery

from .conftest import needs_pg

pytestmark = needs_pg


def test_query_building():
    assert to_or_tsquery("What is the user's favourite editor?") == "user | favourite | editor"
    assert to_or_tsquery("a an the") is None


async def test_save_search_dedupe_and_scopes(pg_db):
    repo, mem = Repo(pg_db), MemoryStore(pg_db)
    s1 = (await repo.create_session("a"))["id"]
    s2 = (await repo.create_session("b"))["id"]
    g = await mem.save(session_id=None, kind="preference", content="The user prefers tabs over spaces in Python projects.", importance=4)
    await mem.save(session_id=s1, kind="project", content="Project uses FastAPI with PostgreSQL and pytest for testing.")
    await mem.save(session_id=s2, kind="project", content="Secret project of session two uses Rust and Axum.")

    hits = await mem.search("which indentation does the user prefer, tabs?", session_id=s1)
    assert hits and hits[0]["id"] == g["id"]
    hits = await mem.search("what testing framework", session_id=s1)
    assert "pytest" in hits[0]["content"]
    # session isolation: s1 must not see s2's memory, global is shared
    assert all("Axum" not in h["content"] for h in await mem.search("Rust Axum project", session_id=s1))
    assert any("Axum" in h["content"] for h in await mem.search("Rust Axum project", session_id=s2))

    again = await mem.save(session_id=None, kind="preference", content="The user prefers tabs over spaces in Python projects!", importance=5)
    assert again.get("deduplicated") and again["id"] == g["id"] and again["importance"] == 5
    assert len(await mem.list(scope="global")) == 1


async def test_prefix_resolve_and_forget_and_consolidate(pg_db):
    mem = MemoryStore(pg_db)
    m = await mem.save(session_id=None, kind="fact", content="Deploy target is Kubernetes on GKE.")
    assert await mem.resolve_prefix(m["id"][:8]) == m["id"]
    assert await mem.forget(m["id"]) and not await mem.forget(m["id"])
    assert (await mem.consolidate())["remaining"] == 0
