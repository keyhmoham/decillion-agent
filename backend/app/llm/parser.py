"""Incremental parser for Agents-A1-4B output.

The chat template's generation prompt ends with ``<think>\\n`` so the model's first tokens are
*reasoning*; it closes with ``</think>`` and then writes the visible answer and/or one or more
``<tool_call>`` blocks.  This parser turns that token stream into typed events and is robust to:

* markers split across stream chunks (``</thi`` + ``nk>``);
* the model re-opening ``<think>`` or placing a ``<tool_call>`` inside its reasoning;
* hallucinated ``<tool_response>`` continuations (generation is flagged as halted);
* truncated tool calls (end of stream while a call is still open).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .toolcall import ParsedToolCall, parse_tool_call_body

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
CALL_OPEN = "<tool_call>"
CALL_CLOSE = "</tool_call>"
HALLUCINATED = "<tool_response>"


class EventKind(str, Enum):
    REASONING = "reasoning"
    CONTENT = "content"
    TOOL_CALL = "tool_call"


@dataclass
class ParserEvent:
    kind: EventKind
    text: str = ""
    call: ParsedToolCall | None = None


class _State(str, Enum):
    THINK = "think"
    CONTENT = "content"
    TOOL = "tool"


def _partial_suffix_len(buf: str, markers: tuple[str, ...]) -> int:
    """Length of the longest suffix of ``buf`` that is a proper prefix of any marker."""
    best = 0
    for marker in markers:
        for n in range(min(len(marker) - 1, len(buf)), 0, -1):
            if buf.endswith(marker[:n]):
                best = max(best, n)
                break
    return best


class StreamParser:
    def __init__(self, start_in_thinking: bool = True) -> None:
        self._state = _State.THINK if start_in_thinking else _State.CONTENT
        self._buf = ""
        self._tool_buf = ""
        self._lstrip_next = True  # swallow leading newlines of each new segment
        self.halted = False  # the model started to hallucinate a tool response
        self.closed_thinking = not start_in_thinking
        self.tool_calls_seen = 0

    # ------------------------------------------------------------------ public API
    @property
    def state(self) -> str:
        return self._state.value

    @property
    def in_thinking(self) -> bool:
        return self._state is _State.THINK

    def feed(self, text: str) -> list[ParserEvent]:
        if self.halted or not text:
            return []
        self._buf += text
        return self._drain(final=False)

    def finish(self) -> list[ParserEvent]:
        events = self._drain(final=True)
        if self._state is _State.TOOL and (self._tool_buf.strip()):
            call = parse_tool_call_body(self._tool_buf)
            call.repaired = True
            if not call.error and not call.name:
                call.error = "Tool call was cut off before a function name was written."
            events.append(ParserEvent(EventKind.TOOL_CALL, call=call))
            self.tool_calls_seen += 1
            self._tool_buf = ""
        return events

    # ------------------------------------------------------------------ internals
    def _emit(self, kind: EventKind, text: str, out: list[ParserEvent]) -> None:
        if not text:
            return
        if self._lstrip_next:
            text = text.lstrip("\r\n")
            if not text:
                return
            self._lstrip_next = False
        out.append(ParserEvent(kind, text))

    def _drain(self, final: bool) -> list[ParserEvent]:
        out: list[ParserEvent] = []
        while True:
            progressed = False
            if self._state is _State.THINK:
                progressed = self._step_think(out, final)
            elif self._state is _State.CONTENT:
                progressed = self._step_content(out, final)
            else:
                progressed = self._step_tool(out, final)
            if not progressed or self.halted:
                break
        return out

    def _find_first(self, markers: tuple[str, ...]) -> tuple[int, str] | None:
        best: tuple[int, str] | None = None
        for marker in markers:
            idx = self._buf.find(marker)
            if idx != -1 and (best is None or idx < best[0]):
                best = (idx, marker)
        return best

    def _flush_safe(self, kind: EventKind, markers: tuple[str, ...], out: list[ParserEvent], final: bool) -> None:
        hold = 0 if final else _partial_suffix_len(self._buf, markers)
        safe, self._buf = (self._buf[: len(self._buf) - hold], self._buf[len(self._buf) - hold:]) if hold else (self._buf, "")
        self._emit(kind, safe, out)

    def _step_think(self, out: list[ParserEvent], final: bool) -> bool:
        markers = (THINK_CLOSE, CALL_OPEN, THINK_OPEN)
        hit = self._find_first(markers)
        if hit is None:
            self._flush_safe(EventKind.REASONING, markers, out, final)
            return False
        idx, marker = hit
        self._emit(EventKind.REASONING, self._buf[:idx], out)
        self._buf = self._buf[idx + len(marker):]
        if marker == THINK_OPEN:
            return True  # redundant re-open of the block we are already in
        if marker == THINK_CLOSE:
            self._state = _State.CONTENT
            self.closed_thinking = True
            self._lstrip_next = True
        else:  # tool call opened inside reasoning (recoverable)
            self._state = _State.TOOL
            self._tool_buf = ""
            self.closed_thinking = True
        return True

    def _step_content(self, out: list[ParserEvent], final: bool) -> bool:
        markers = (CALL_OPEN, THINK_OPEN, THINK_CLOSE, HALLUCINATED)
        hit = self._find_first(markers)
        if hit is None:
            self._flush_safe(EventKind.CONTENT, markers, out, final)
            return False
        idx, marker = hit
        self._emit(EventKind.CONTENT, self._buf[:idx], out)
        self._buf = self._buf[idx + len(marker):]
        if marker == CALL_OPEN:
            self._state = _State.TOOL
            self._tool_buf = ""
        elif marker == THINK_OPEN:
            self._state = _State.THINK
            self._lstrip_next = True
        elif marker == THINK_CLOSE:
            self._lstrip_next = True  # stray close tag: drop it
        else:  # HALLUCINATED
            self.halted = True
            self._buf = ""
            return False
        return True

    def _step_tool(self, out: list[ParserEvent], final: bool) -> bool:
        idx = self._buf.find(CALL_CLOSE)
        if idx == -1:
            hold = 0 if final else _partial_suffix_len(self._buf, (CALL_CLOSE,))
            cut = len(self._buf) - hold
            self._tool_buf += self._buf[:cut]
            self._buf = self._buf[cut:]
            # A new <tool_call> before the old one closed: treat the previous as truncated.
            return False
        self._tool_buf += self._buf[:idx]
        self._buf = self._buf[idx + len(CALL_CLOSE):]
        call = parse_tool_call_body(self._tool_buf)
        out.append(ParserEvent(EventKind.TOOL_CALL, call=call))
        self.tool_calls_seen += 1
        self._tool_buf = ""
        self._state = _State.CONTENT
        self._lstrip_next = True
        return True


def parse_completion(text: str, *, start_in_thinking: bool = True) -> tuple[str, str, list[ParsedToolCall]]:
    """Convenience: parse a complete generation -> (reasoning, content, tool_calls)."""
    parser = StreamParser(start_in_thinking)
    events = parser.feed(text) + parser.finish()
    reasoning, content, calls = [], [], []
    for ev in events:
        if ev.kind is EventKind.REASONING:
            reasoning.append(ev.text)
        elif ev.kind is EventKind.CONTENT:
            content.append(ev.text)
        elif ev.call is not None:
            calls.append(ev.call)
    return "".join(reasoning).strip(), "".join(content).strip(), calls
