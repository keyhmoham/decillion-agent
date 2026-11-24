"""LangChain chat model dedicated to Agents-A1-4B-Kimi-heretic.

Highlights (all of them are about matching the model's *trained* behaviour):

* renders the exact upstream chat template and uses raw completions (see ``template.py``);
* parses the native ``<think>`` / ``<tool_call><function=..>`` stream incrementally;
* coerces XML string parameters to the JSON-schema types of the bound tools;
* enforces a **thinking budget** and detects degenerate repetition inside ``<think>`` - when
  triggered the block is closed for the model and generation continues;
* falls back to the model's native *no-think* mode when a turn would otherwise come back empty;
* supports ``tool_choice=<name>`` by pre-filling ``<tool_call><function=name>`` (guaranteed
  structured output for planners / extractors);
* exposes ``reasoning_content`` on messages so the template can replay reasoning for the
  current agentic turn exactly as in training.
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, ConfigDict, Field

from .backends import GenerationRequest, LLMBackend
from .parser import EventKind, StreamParser
from .profile import DEFAULT_PROFILE, A1Profile, SamplingParams
from .template import render_prompt
from .tokens import TokenCounter, get_counter
from .toolcall import coerce_arguments, compact_tool_schema, resolve_tool_name

THINK_CLOSE_NUDGE = "\n\nI have reasoned enough. I must now act on my conclusions.\n</think>\n\n"


# --------------------------------------------------------------------------------------------
# message conversion
# --------------------------------------------------------------------------------------------

def _flatten_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            if block.get("type") in (None, "text") and "text" in block:
                parts.append(str(block["text"]))
    return "".join(parts)


def to_template_messages(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    """LangChain messages -> dicts accepted by the model's chat template."""
    out: list[dict[str, Any]] = []
    system_chunks: list[str] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            text = _flatten_content(m.content).strip()
            if not out:  # leading system message(s) are merged
                system_chunks.append(text)
            else:  # the template only allows a system message at index 0
                out.append({"role": "user", "content": f"[System note]\n{text}"})
            continue
        if system_chunks and (not out or out[0].get("role") != "system"):
            out.insert(0, {"role": "system", "content": "\n\n".join(system_chunks)})
            system_chunks = []
        if isinstance(m, HumanMessage):
            out.append({"role": "user", "content": _flatten_content(m.content)})
        elif isinstance(m, AIMessage):
            msg: dict[str, Any] = {"role": "assistant", "content": _flatten_content(m.content)}
            reasoning = m.additional_kwargs.get("reasoning_content")
            if reasoning:
                msg["reasoning_content"] = reasoning
            if m.tool_calls:
                msg["tool_calls"] = [
                    {"function": {"name": tc["name"], "arguments": tc.get("args", {})}} for tc in m.tool_calls
                ]
            out.append(msg)
        elif isinstance(m, ToolMessage):
            out.append({"role": "tool", "content": _flatten_content(m.content)})
        else:
            out.append({"role": "user", "content": _flatten_content(m.content)})
    if system_chunks:
        out.insert(0, {"role": "system", "content": "\n\n".join(system_chunks)})
    # The template needs at least one genuine user query.
    if not any(
        m["role"] == "user" and not (m["content"].strip().startswith("<tool_response>"))
        for m in out
    ):
        out.append({"role": "user", "content": "Continue."})
    return out


# --------------------------------------------------------------------------------------------
# degenerate-repetition detector (4B models occasionally loop inside <think>)
# --------------------------------------------------------------------------------------------

def looks_degenerate(text: str, window: int = 160, min_repeats: int = 3) -> bool:
    if len(text) < window * min_repeats:
        return False
    tail = text[-window:]
    if not tail.strip():
        return False
    return text.count(tail) >= min_repeats


# --------------------------------------------------------------------------------------------
# generation state shared between phases
# --------------------------------------------------------------------------------------------

@dataclass
class _Gen:
    raw: str = ""
    reasoning_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None
    forced_close: bool = False
    degenerate: bool = False
    fallback_nothink: bool = False
    n_content: int = 0
    n_calls: int = 0
    n_errors: int = 0
    tool_index: int = 0
    reasoning_text: list[str] = field(default_factory=list)


class A1ChatModel(BaseChatModel):
    """Chat model for ``nutboy02/Agents-A1-4B-Kimi-heretic``."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    backend: Any = Field(exclude=True)  # LLMBackend
    profile: A1Profile = DEFAULT_PROFILE
    sampling: SamplingParams = Field(default_factory=SamplingParams)
    enable_thinking: bool = True
    thinking_budget: int = 1800
    max_tokens: int = 4096
    empty_retries: int = 1
    counter: Any = Field(default=None, exclude=True)  # TokenCounter
    name_hint: str = "agents-a1-4b"

    # -------------------------------------------------------------------------- LangChain API
    @property
    def _llm_type(self) -> str:
        return "agents-a1-4b-kimi-heretic"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.profile.ollama_tag, "thinking": self.enable_thinking}

    def bind_tools(  # type: ignore[override]
        self,
        tools: Sequence[dict[str, Any] | type | BaseTool | Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable:
        formatted = [compact_tool_schema(convert_to_openai_tool(t)) for t in tools]
        return self.bind(tools=formatted, tool_choice=tool_choice, **kwargs)

    def with_options(self, **overrides: Any) -> Runnable:
        """Bind per-call overrides (enable_thinking, thinking_budget, max_tokens, temperature)."""
        return self.bind(**overrides)

    # -------------------------------------------------------------------------- token counting
    @property
    def _counter(self) -> TokenCounter:
        return self.counter or get_counter()

    def count_prompt_tokens(self, messages: Sequence[BaseMessage], tools: Sequence[Any] | None = None) -> int:
        formatted = [convert_to_openai_tool(t) for t in tools] if tools else None
        prompt = render_prompt(to_template_messages(messages), formatted, enable_thinking=self.enable_thinking)
        return self._counter.count(prompt)

    def get_num_tokens(self, text: str) -> int:
        return self._counter.count(text)

    def get_num_tokens_from_messages(self, messages: list[BaseMessage], tools: Sequence[Any] | None = None, **_: Any) -> int:  # type: ignore[override]
        return self.count_prompt_tokens(messages, tools)

    # -------------------------------------------------------------------------- generation
    def _stop(self, stop: list[str] | None) -> list[str]:
        merged = list(self.profile.stop)
        for s in stop or []:
            if s not in merged:
                merged.append(s)
        return merged

    def _chunk(self, *, content: str = "", reasoning: str = "", **extra: Any) -> ChatGenerationChunk:
        kwargs: dict[str, Any] = {}
        if reasoning:
            kwargs["reasoning_content"] = reasoning
        msg = AIMessageChunk(content=content, additional_kwargs=kwargs, **extra)
        return ChatGenerationChunk(message=msg)

    async def _phase(
        self,
        *,
        prompt: str,
        parser: StreamParser,
        sampling: SamplingParams,
        max_tokens: int,
        stop: list[str],
        budget: int | None,
        gen: _Gen,
        tools: list[dict[str, Any]],
        tool_names: list[str],
        schemas: dict[str, dict[str, Any]],
    ) -> AsyncIterator[ChatGenerationChunk]:
        """One backend call. Sets ``gen.forced_close`` when it aborts a runaway <think>."""
        req = GenerationRequest(prompt=prompt, sampling=sampling, max_tokens=max_tokens, stop=stop)
        stream = self.backend.stream(req)
        raw_phase = ""
        aborted = False
        reported_prompt = reported_completion = False
        try:
            async for piece in stream:
                if piece.prompt_tokens:
                    gen.prompt_tokens += piece.prompt_tokens
                    reported_prompt = True
                if piece.completion_tokens:
                    gen.completion_tokens += piece.completion_tokens
                    reported_completion = True
                if piece.finish_reason:
                    gen.finish_reason = piece.finish_reason
                if not piece.text:
                    continue
                was_thinking = parser.in_thinking
                raw_phase += piece.text
                gen.raw += piece.text
                if was_thinking:
                    gen.reasoning_tokens += 1
                for ev in parser.feed(piece.text):
                    for chunk in self._events_to_chunks([ev], gen, tool_names, schemas):
                        yield chunk
                if parser.halted:
                    aborted = True
                    break
                if budget is not None and parser.in_thinking:
                    over = gen.reasoning_tokens >= budget
                    loop = gen.reasoning_tokens % 48 == 0 and looks_degenerate(raw_phase)
                    if over or loop:
                        gen.forced_close = True
                        gen.degenerate = loop
                        aborted = True
                        break
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                await aclose()
            # Some servers (llama-cpp-python) do not report usage while streaming: estimate it.
            if not reported_prompt:
                gen.prompt_tokens += self._counter.count(prompt)
            if not reported_completion:
                gen.completion_tokens += self._counter.count(raw_phase)
        if not aborted or parser.halted:
            for ev in parser.finish():
                for chunk in self._events_to_chunks([ev], gen, tool_names, schemas):
                    yield chunk

    def _events_to_chunks(
        self,
        events: list[Any],
        gen: _Gen,
        tool_names: list[str],
        schemas: dict[str, dict[str, Any]],
    ) -> list[ChatGenerationChunk]:
        chunks: list[ChatGenerationChunk] = []
        for ev in events:
            if ev.kind is EventKind.REASONING:
                gen.reasoning_text.append(ev.text)
                chunks.append(self._chunk(reasoning=ev.text))
            elif ev.kind is EventKind.CONTENT:
                gen.n_content += len(ev.text.strip())
                chunks.append(self._chunk(content=ev.text))
            elif ev.kind is EventKind.TOOL_CALL and ev.call is not None:
                call = ev.call
                if call.error or not call.name:
                    gen.n_errors += 1
                    msg = AIMessageChunk(
                        content="",
                        additional_kwargs={"tool_call_errors": [{"raw": call.raw.strip(), "error": call.error or "unknown"}]},
                    )
                    chunks.append(ChatGenerationChunk(message=msg))
                    continue
                name = resolve_tool_name(call.name, tool_names) or call.name
                args = coerce_arguments(call.arguments, schemas.get(name))
                gen.n_calls += 1
                msg = AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": name,
                            "args": json.dumps(args, ensure_ascii=False),
                            "id": f"call_{uuid.uuid4().hex[:10]}",
                            "index": gen.tool_index,
                        }
                    ],
                )
                gen.tool_index += 1
                chunks.append(ChatGenerationChunk(message=msg))
        return chunks

    async def _astream(  # type: ignore[override]
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        tools: list[dict[str, Any]] = kwargs.get("tools") or []
        tool_choice = kwargs.get("tool_choice")
        think = bool(kwargs.get("enable_thinking", self.enable_thinking))
        budget = kwargs.get("thinking_budget", self.thinking_budget)
        max_tokens = int(kwargs.get("max_tokens", self.max_tokens))
        sampling = self.sampling.with_(
            temperature=kwargs.get("temperature"),
            top_p=kwargs.get("top_p"),
            top_k=kwargs.get("top_k"),
        )
        tool_names = [t["function"]["name"] for t in tools]
        schemas = {t["function"]["name"]: t["function"].get("parameters") or {} for t in tools}
        forced = tool_choice if isinstance(tool_choice, str) and tool_choice in tool_names else None
        if forced:
            think = False  # structured calls are made in the model's native no-think mode
        stops = self._stop(stop)
        tmpl_messages = to_template_messages(messages)
        gen = _Gen()

        for attempt in range(self.empty_retries + 1):
            attempt_think = think and attempt == 0
            gen.fallback_nothink = attempt > 0
            prompt = render_prompt(tmpl_messages, tools or None, enable_thinking=attempt_think)
            parser = StreamParser(start_in_thinking=attempt_think)
            if forced:
                prefill = f"<tool_call>\n<function={forced}>\n"
                prompt += prefill
                parser.feed(prefill)
            sp = sampling if attempt == 0 else sampling.with_(temperature=min(sampling.temperature, 0.3))

            before = (gen.n_content, gen.n_calls, gen.n_errors)
            async for ch in self._phase(
                prompt=prompt, parser=parser, sampling=sp, max_tokens=max_tokens, stop=stops,
                budget=budget if attempt_think else None, gen=gen, tools=tools,
                tool_names=tool_names, schemas=schemas,
            ):
                yield ch

            if gen.forced_close and parser.in_thinking:
                # Phase 2: close the runaway <think> block ourselves and let the model answer.
                reasoning_so_far = "".join(gen.reasoning_text)
                prompt2 = prompt + reasoning_so_far.rstrip() + THINK_CLOSE_NUDGE
                parser2 = StreamParser(start_in_thinking=False)
                remaining = max(256, max_tokens - gen.reasoning_tokens)
                async for ch in self._phase(
                    prompt=prompt2, parser=parser2, sampling=sampling.with_(temperature=min(sampling.temperature, 0.4)),
                    max_tokens=remaining, stop=stops, budget=None, gen=gen, tools=tools,
                    tool_names=tool_names, schemas=schemas,
                ):
                    yield ch

            produced = (gen.n_content, gen.n_calls, gen.n_errors) != before
            if produced or attempt >= self.empty_retries or forced:
                break
            # empty turn -> retry in the model's native no-think mode

        usage = {
            "input_tokens": gen.prompt_tokens,
            "output_tokens": gen.completion_tokens,
            "total_tokens": gen.prompt_tokens + gen.completion_tokens,
        }
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                usage_metadata=usage,  # type: ignore[arg-type]
                response_metadata={
                    "finish_reason": gen.finish_reason,
                    "thinking_forced_closed": gen.forced_close,
                    "thinking_degenerate": gen.degenerate,
                    "nothink_fallback": gen.fallback_nothink,
                    "reasoning_tokens": gen.reasoning_tokens,
                    "model": self.profile.ollama_tag,
                },
            )
        )

    async def _agenerate(  # type: ignore[override]
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        final: ChatGenerationChunk | None = None
        async for chunk in self._astream(messages, stop, run_manager, **kwargs):
            if run_manager is not None and (chunk.text or chunk.message.additional_kwargs or chunk.message.tool_call_chunks):
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            final = chunk if final is None else final + chunk
        assert final is not None
        msg = final.message
        ai = AIMessage(
            content=_flatten_content(msg.content).strip(),
            additional_kwargs={
                **{k: (v.strip() if isinstance(v, str) else v) for k, v in msg.additional_kwargs.items()},
            },
            tool_calls=[
                {"name": tc["name"], "args": tc["args"], "id": tc["id"], "type": "tool_call"}
                for tc in getattr(msg, "tool_calls", [])
            ],
            usage_metadata=getattr(msg, "usage_metadata", None),
            response_metadata=msg.response_metadata,
        )
        return ChatResult(generations=[ChatGeneration(message=ai)])

    def _generate(  # type: ignore[override]
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        coro = self._agenerate(messages, stop, None, **kwargs)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()

    # -------------------------------------------------------------------------- structured output
    async def aextract(
        self,
        messages: Sequence[BaseMessage],
        schema: type[BaseModel],
        *,
        name: str | None = None,
        max_tokens: int = 1200,
    ) -> BaseModel:
        """Force a single tool call that matches ``schema`` and return the validated instance.

        Uses the model's native no-think mode with a pre-filled ``<function=...>`` header, which
        is the most reliable way to get structured data from a 4B model.
        """
        tool = convert_to_openai_tool(schema)
        fname = name or tool["function"]["name"]
        tool["function"]["name"] = fname
        runnable = self.bind(tools=[tool], tool_choice=fname, max_tokens=max_tokens)
        last_err: Exception | None = None
        for _ in range(2):
            ai = await runnable.ainvoke(list(messages))
            for tc in ai.tool_calls:
                if tc["name"] == fname:
                    try:
                        return schema.model_validate(tc["args"])
                    except Exception as exc:  # noqa: BLE001
                        last_err = exc
            if last_err is None:
                last_err = ValueError("model did not produce the requested structured call")
        raise ValueError(f"structured extraction failed: {last_err}")

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable:  # type: ignore[override]
        from langchain_core.output_parsers.openai_tools import PydanticToolsParser

        tool = convert_to_openai_tool(schema)
        name = tool["function"]["name"]
        bound = self.bind_tools([schema], tool_choice=name)
        return bound | PydanticToolsParser(tools=[schema], first_tool_only=True)


_SENT_END = re.compile(r"(?<=[.!?])\s")


def build_chat_model(settings: Any, backend: LLMBackend, **overrides: Any) -> A1ChatModel:
    from .profile import sampling_from_settings

    params: dict[str, Any] = dict(
        backend=backend,
        sampling=sampling_from_settings(settings),
        enable_thinking=settings.enable_thinking,
        thinking_budget=settings.thinking_budget,
        max_tokens=settings.llm_max_tokens,
        counter=get_counter(settings.tokenizer_path),
    )
    params.update(overrides)
    return A1ChatModel(**params)
