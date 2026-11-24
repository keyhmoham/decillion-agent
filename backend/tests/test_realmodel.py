"""Live compatibility checks against a running Agents-A1-4B server.

Run with e.g.::

    A1_TEST_BACKEND=openai A1_TEST_BASE_URL=http://127.0.0.1:8081 pytest -m realmodel
"""

import os

import pytest

from app.config import Settings
from app.llm.backends import make_backend
from app.llm.chat_model import build_chat_model
from app.llm.selftest import CHECKS, run_selftest

BASE = os.environ.get("A1_TEST_BASE_URL")
pytestmark = [pytest.mark.realmodel, pytest.mark.skipif(not BASE, reason="A1_TEST_BASE_URL not set")]


@pytest.fixture(scope="module")
def live_model():
    s = Settings(llm_backend=os.environ.get("A1_TEST_BACKEND", "openai"), llm_base_url=BASE or "",
                 llm_model=os.environ.get("A1_TEST_MODEL", "a1"), thinking_budget=1200, llm_max_tokens=1500)
    return build_chat_model(s, make_backend(s))


@pytest.mark.parametrize("name", [n for n, _ in CHECKS])
async def test_live(live_model, name):
    (res,) = await run_selftest(live_model, only=[name])
    assert res.passed, res.detail
