"""Exact prompt rendering for Agents-A1-4B (Qwen3.5 hybrid, ChatML + XML tool calls).

Why this module exists
----------------------
The Ollama package `nutboy02/Agents-A1-4B-Kimi-heretic` ships a *legacy* TEMPLATE that does not
match what the model was trained on:

* it uses ``<thought>`` tags, while the model is trained on ``<think>`` / ``</think>``;
* it dumps tools as bare JSON lines, without the ``# Tools`` preamble and without the
  ``<tool_call><function=...><parameter=...>`` calling convention;
* it never renders previous tool calls or tool results (``<tool_response>``);
* it adds ``</thought>`` as a stop sequence.

Everything the model learned (thinking, tool calling, tool results) lives in the chat template
embedded in the GGUF / HF repo (identical for ``InternScience/Agents-A1-4B`` and
``hotdogs/Agents-A1-4B-kimi-Preview-heretic``).  We therefore render that template ourselves and
talk to the serving engine in *raw completion* mode.  The template file is vendored verbatim in
``chat_template.jinja``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import StrictUndefined  # noqa: F401  (kept for debugging variants)
from jinja2.sandbox import ImmutableSandboxedEnvironment

TEMPLATE_PATH = Path(__file__).with_name("chat_template.jinja")

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
TOOL_RESPONSE_OPEN = "<tool_response>"

# What the template appends when ``add_generation_prompt`` is true:
GEN_PROMPT_THINK = f"{IM_START}assistant\n{THINK_OPEN}\n"
GEN_PROMPT_NOTHINK = f"{IM_START}assistant\n{THINK_OPEN}\n\n{THINK_CLOSE}\n\n"


class TemplateError(ValueError):
    """Raised when a conversation cannot be rendered with the model template."""


def _raise_exception(message: str) -> None:
    raise TemplateError(message)


def _tojson(value: Any, ensure_ascii: bool = False, indent: int | None = None,
            separators: tuple[str, str] | None = None, sort_keys: bool = False) -> str:
    """Mirror of ``transformers``' chat-template ``tojson`` (no HTML escaping, UTF-8 friendly)."""
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators,
                      sort_keys=sort_keys)


@lru_cache(maxsize=1)
def _environment() -> ImmutableSandboxedEnvironment:
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise_exception
    return env


@lru_cache(maxsize=1)
def _compiled():
    return _environment().from_string(TEMPLATE_PATH.read_text(encoding="utf-8"))


def _normalise_tool(tool: Mapping[str, Any]) -> dict[str, Any]:
    """Tools are rendered as OpenAI-style ``{"type": "function", "function": {...}}`` objects."""
    if "function" in tool and tool.get("type") == "function":
        return dict(tool)
    if "name" in tool:
        return {"type": "function", "function": dict(tool)}
    raise TemplateError(f"Unsupported tool definition: {tool!r}")


def _normalise_message(msg: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"role": msg["role"], "content": msg.get("content") or ""}
    if msg.get("reasoning_content"):
        out["reasoning_content"] = msg["reasoning_content"]
    if msg.get("tool_calls"):
        calls = []
        for tc in msg["tool_calls"]:
            fn = tc.get("function", tc)
            args = fn.get("arguments", tc.get("args", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    args = {"input": args}
            calls.append({"function": {"name": fn["name"], "arguments": args or {}}})
        out["tool_calls"] = calls
    return out


def render_prompt(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None = None,
    *,
    add_generation_prompt: bool = True,
    enable_thinking: bool = True,
) -> str:
    """Render ``messages`` exactly as ``tokenizer.apply_chat_template`` would.

    ``enable_thinking=False`` makes the template pre-fill an empty ``<think>`` block, which is
    the model's native "no reasoning" mode.
    """
    msgs = [_normalise_message(m) for m in messages]
    tool_defs = [_normalise_tool(t) for t in tools] if tools else None
    try:
        return _compiled().render(
            messages=msgs,
            tools=tool_defs,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
    except TemplateError:
        raise
    except Exception as exc:  # jinja errors are unhelpful, wrap with context
        raise TemplateError(f"Failed to render chat template: {exc}") from exc


def generation_prefix(enable_thinking: bool) -> str:
    return GEN_PROMPT_THINK if enable_thinking else GEN_PROMPT_NOTHINK
