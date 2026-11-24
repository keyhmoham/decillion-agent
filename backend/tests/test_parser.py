import itertools

import pytest

from app.llm.parser import EventKind, StreamParser, parse_completion

CALL = """<tool_call>
<function=write_file>
<parameter=path>
/workspace/a.py
</parameter>
<parameter=content>
print("hi")

</parameter>
</function>
</tool_call>"""

SAMPLE = "Plan: write the file.\n</think>\n\nNow I'll create it:" + CALL


def collect(text, start_thinking=True, split=None):
    p = StreamParser(start_thinking)
    parts = [text] if split is None else [text[: split[0]], text[split[0] : split[1]], text[split[1] :]]
    events = []
    for part in parts:
        events += p.feed(part)
    events += p.finish()
    reasoning = "".join(e.text for e in events if e.kind is EventKind.REASONING)
    content = "".join(e.text for e in events if e.kind is EventKind.CONTENT)
    calls = [e.call for e in events if e.kind is EventKind.TOOL_CALL]
    return reasoning, content, calls


def test_basic_think_content_call():
    reasoning, content, calls = collect(SAMPLE)
    assert reasoning.strip() == "Plan: write the file."
    assert content.strip() == "Now I'll create it:"
    assert len(calls) == 1 and calls[0].name == "write_file"
    assert calls[0].arguments == {"path": "/workspace/a.py", "content": 'print("hi")\n'}


def test_every_chunk_boundary_gives_identical_result():
    ref = collect(SAMPLE)
    n = len(SAMPLE)
    for i, j in itertools.combinations(range(1, n), 2):
        r = collect(SAMPLE, split=(i, j))
        assert (r[0], r[1]) == (ref[0], ref[1]), (i, j)
        assert [c.arguments for c in r[2]] == [c.arguments for c in ref[2]], (i, j)


def test_no_think_mode_starts_in_content():
    _, content, calls = collect("Hello there", start_thinking=False)
    assert content == "Hello there" and not calls


def test_model_reopening_think_block_is_tolerated():
    reasoning, content, _ = collect("<think>\nabc\n</think>\n\nanswer")
    assert reasoning.strip() == "abc" and content == "answer"


def test_tool_call_inside_reasoning_is_recovered():
    reasoning, content, calls = collect("I will call it.\n" + CALL)
    assert calls and calls[0].name == "write_file"
    assert "I will call it." in reasoning and content == ""


def test_parallel_calls():
    text = "x</think>\n\n" + CALL + "\n" + CALL.replace("a.py", "b.py")
    _, _, calls = collect(text)
    assert [c.arguments["path"] for c in calls] == ["/workspace/a.py", "/workspace/b.py"]


def test_hallucinated_tool_response_halts_generation():
    p = StreamParser(True)
    events = p.feed("r</think>\n\n" + CALL + "\n<tool_response>\nfake\n</tool_response>")
    assert p.halted
    assert [e.kind for e in events].count(EventKind.TOOL_CALL) == 1
    assert "fake" not in "".join(e.text for e in events)


def test_truncated_tool_call_is_surfaced_not_lost():
    _, _, calls = collect("x</think>\n\n<tool_call>\n<function=bash>\n<parameter=command>\nls -la")
    assert calls and calls[0].name == "bash" and calls[0].arguments["command"].startswith("ls -la")
    assert calls[0].repaired


def test_unfinished_think_is_reasoning_only():
    r, c, calls = collect("still thinking...")
    assert r == "still thinking..." and c == "" and not calls


def test_partial_marker_lookalike_not_swallowed():
    r, c, _ = collect("a < b and x<y</think>ok")
    assert r.strip() == "a < b and x<y" and c == "ok"


def test_parse_completion_helper():
    r, c, calls = parse_completion(SAMPLE)
    assert r and c and len(calls) == 1
