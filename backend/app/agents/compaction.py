"""Context management for a model fine-tuned on short traces.

Two tiers:
1. *micro-compaction*: shrink large old tool outputs in place (cheap, lossless enough).
2. *summarisation*: replace the old part of the conversation by a structured summary written by
   the model itself, keeping the most recent turns verbatim (tool-call/result pairs stay intact).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from ..llm.chat_model import A1ChatModel
from ..tools.base import truncate_middle
from .prompts import COMPACT_PROMPT

SUMMARY_MARK = "summary"


def is_summary(m: BaseMessage) -> bool:
    return bool(m.additional_kwargs.get(SUMMARY_MARK))


def microcompact(messages: list[BaseMessage], keep_recent: int = 8, max_chars: int = 1200) -> list[ToolMessage]:
    """Return replacement ToolMessages (same ids) for old, oversized tool results."""
    out: list[ToolMessage] = []
    cutoff = max(0, len(messages) - keep_recent)
    for m in messages[:cutoff]:
        if isinstance(m, ToolMessage) and isinstance(m.content, str) and len(m.content) > max_chars and not m.additional_kwargs.get("trimmed"):
            out.append(ToolMessage(
                content=truncate_middle(m.content, max_chars, "old tool output trimmed to save context"),
                tool_call_id=m.tool_call_id, name=m.name, id=m.id, additional_kwargs={**m.additional_kwargs, "trimmed": True},
            ))
    return out


def split_for_compaction(messages: list[BaseMessage], keep_last: int) -> tuple[list[BaseMessage], list[BaseMessage]]:
    """Split so the tail never starts inside a tool-call/tool-result group."""
    if len(messages) <= keep_last + 1:
        return [], messages
    idx = len(messages) - keep_last
    while idx > 0 and isinstance(messages[idx], ToolMessage):
        idx -= 1
    return messages[:idx], messages[idx:]


def transcript(messages: list[BaseMessage], tool_chars: int = 500, ai_chars: int = 1500) -> str:
    lines: list[str] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(("SUMMARY: " if is_summary(m) else "USER: ") + str(m.content)[:3000])
        elif isinstance(m, AIMessage):
            calls = "; ".join(f"{tc['name']}({', '.join(f'{k}={str(v)[:80]!r}' for k, v in tc['args'].items())})" for tc in m.tool_calls)
            text = str(m.content)[:ai_chars]
            lines.append(f"ASSISTANT: {text}{' [calls: ' + calls + ']' if calls else ''}")
        elif isinstance(m, ToolMessage):
            lines.append(f"TOOL RESULT ({m.name}): {truncate_middle(str(m.content), tool_chars)}")
    return "\n".join(lines)


@dataclass
class CompactionResult:
    messages: list[BaseMessage]
    summary: str
    tokens_before: int
    tokens_after: int


async def summarize(model: A1ChatModel, messages: list[BaseMessage], config: dict[str, Any] | None = None, chunk_chars: int = 24000) -> str:
    text = transcript(messages)
    running = ""
    parts = [text[i : i + chunk_chars] for i in range(0, len(text), chunk_chars)] or [""]
    cfg = {**(config or {}), "tags": ["internal"], "metadata": {**((config or {}).get("metadata") or {}), "purpose": "compacting context"}}
    for part in parts:
        prompt = f"{COMPACT_PROMPT}\n\n" + (f"Existing summary so far:\n{running}\n\n" if running else "") + f"New conversation excerpt:\n{part}"
        ai = await model.bind(enable_thinking=False, max_tokens=900, temperature=0.3).ainvoke([HumanMessage(content=prompt)], cfg)
        running = str(ai.content).strip() or running
    return running


async def compact(
    model: A1ChatModel, messages: list[BaseMessage], *, count_tokens, keep_last: int = 6, config: dict[str, Any] | None = None,
) -> CompactionResult | None:
    head, tail = split_for_compaction(messages, keep_last)
    if not head:
        return None
    before = count_tokens(messages)
    last_user = next((m for m in reversed(messages) if isinstance(m, HumanMessage) and not is_summary(m)), None)
    summary = await summarize(model, head, config)
    if not summary:
        return None
    body = "[Summary of the earlier conversation. Treat it as ground truth for what has already happened.]\n"
    if last_user is not None and last_user not in tail:
        body += f"## User request (verbatim)\n{str(last_user.content)[:2500]}\n\n"
    body += summary
    summary_msg = HumanMessage(content=body, additional_kwargs={SUMMARY_MARK: True})
    new = [summary_msg, *tail]
    return CompactionResult(new, summary, before, count_tokens(new))
