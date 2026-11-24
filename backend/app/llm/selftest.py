"""Model-compatibility self-test.

Runs a small battery of checks against the *live* model server to verify that the whole stack
(template -> raw completion -> stream parser -> typed tool calls) behaves as the model expects.
Used by ``scripts/verify_model.py``, the ``/api/model/selftest`` endpoint and the ``realmodel``
pytest marker.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .chat_model import A1ChatModel

SYSTEM = "You are a precise assistant with access to tools. Call a tool only when it is needed."


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny, 21C in {city}"


@tool
def add_numbers(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b


@tool
def search_notes(query: str, tags: list[str], limit: int = 5, exact: bool = False) -> str:
    """Search the user's notes. `tags` is a list of tag strings."""
    return "[]"


class Ticket(BaseModel):
    """A support ticket."""

    title: str = Field(description="Short title")
    priority: int = Field(description="1 (low) to 5 (urgent)")
    labels: list[str] = Field(description="Labels")


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    seconds: float


async def _no_tool_needed(m: A1ChatModel) -> str:
    ai = await m.bind_tools([get_weather, add_numbers]).ainvoke(
        [SystemMessage(content=SYSTEM), HumanMessage(content="What is the capital of France? One word.")]
    )
    assert not ai.tool_calls, f"unexpected tool call {ai.tool_calls}"
    assert "paris" in ai.content.lower(), f"answer was {ai.content!r}"
    return f"answered directly: {ai.content[:60]!r}"


async def _single_tool_call(m: A1ChatModel) -> str:
    ai = await m.bind_tools([get_weather, add_numbers]).ainvoke(
        [SystemMessage(content=SYSTEM), HumanMessage(content="What's the weather in Paris right now?")]
    )
    assert ai.tool_calls, f"no tool call; content={ai.content!r}"
    call = ai.tool_calls[0]
    assert call["name"] == "get_weather", call
    assert "paris" in str(call["args"].get("city", "")).lower(), call
    return f"{call['name']}({call['args']})"


async def _typed_arguments(m: A1ChatModel) -> str:
    ai = await m.bind_tools([search_notes, add_numbers]).ainvoke(
        [
            SystemMessage(content=SYSTEM),
            HumanMessage(content="Search my notes for 'budget' with tags work and finance, limit 3."),
        ]
    )
    assert ai.tool_calls, f"no tool call; content={ai.content!r}"
    args = ai.tool_calls[0]["args"]
    assert ai.tool_calls[0]["name"] == "search_notes", ai.tool_calls
    assert isinstance(args.get("tags"), list) and args["tags"], args
    assert isinstance(args.get("limit", 5), int), args
    return str(args)


async def _tool_result_followup(m: A1ChatModel) -> str:
    history = [
        SystemMessage(content=SYSTEM),
        HumanMessage(content="Add 1234 and 4321 using the tool and tell me the sum."),
        AIMessage(content="", tool_calls=[{"name": "add_numbers", "args": {"a": 1234, "b": 4321}, "id": "c1"}],
                  additional_kwargs={"reasoning_content": "I need to add the numbers with the tool."}),
        ToolMessage(content="5555", tool_call_id="c1", name="add_numbers"),
    ]
    ai = await m.bind_tools([add_numbers]).ainvoke(history)
    assert not ai.tool_calls, f"looped on tools: {ai.tool_calls}"
    assert "5555" in ai.content.replace(",", ""), f"answer was {ai.content!r}"
    return ai.content[:80]


async def _structured_extraction(m: A1ChatModel) -> str:
    t = await m.aextract(
        [HumanMessage(content="Customer says the checkout page crashes on payment, they are furious, "
                              "this is blocking all sales. Make a ticket with labels checkout and payments.")],
        Ticket, name="create_ticket",
    )
    assert isinstance(t, Ticket) and t.title and 1 <= t.priority <= 5 and t.labels, t
    return t.model_dump_json()


async def _thinking_toggle(m: A1ChatModel) -> str:
    on = await m.bind(enable_thinking=True).ainvoke([HumanMessage(content="Is 91 prime? Answer yes or no.")])
    off = await m.bind(enable_thinking=False).ainvoke([HumanMessage(content="Is 91 prime? Answer yes or no.")])
    assert on.additional_kwargs.get("reasoning_content"), "thinking mode produced no reasoning"
    assert not off.additional_kwargs.get("reasoning_content"), "no-think mode still reasoned"
    assert "no" in off.content.lower(), off.content
    return f"think={len(on.additional_kwargs['reasoning_content'])} chars, nothink answer={off.content[:30]!r}"


CHECKS: list[tuple[str, Callable[[A1ChatModel], Awaitable[str]]]] = [
    ("direct answer without tools", _no_tool_needed),
    ("single tool call", _single_tool_call),
    ("typed arguments (array/int)", _typed_arguments),
    ("tool result follow-up", _tool_result_followup),
    ("structured extraction", _structured_extraction),
    ("thinking toggle", _thinking_toggle),
]


async def run_selftest(model: A1ChatModel, only: list[str] | None = None) -> list[CheckResult]:
    results: list[CheckResult] = []
    for name, fn in CHECKS:
        if only and name not in only:
            continue
        t0 = time.time()
        try:
            detail = await fn(model)
            results.append(CheckResult(name, True, detail, round(time.time() - t0, 1)))
        except Exception as exc:  # noqa: BLE001
            results.append(CheckResult(name, False, f"{type(exc).__name__}: {exc}", round(time.time() - t0, 1)))
    return results


def results_to_dicts(results: list[CheckResult]) -> list[dict]:
    return [asdict(r) for r in results]
