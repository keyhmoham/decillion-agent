import pytest

from app.llm.template import GEN_PROMPT_NOTHINK, GEN_PROMPT_THINK, TemplateError, render_prompt

TOOLS = [
    {"type": "function", "function": {"name": "bash", "description": "Run a shell command",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}, "required": ["command"]}}},
]


def test_generation_prompt_think_and_nothink():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}]
    assert render_prompt(msgs, enable_thinking=True).endswith(GEN_PROMPT_THINK)
    assert render_prompt(msgs, enable_thinking=False).endswith(GEN_PROMPT_NOTHINK)


def test_tools_block_and_native_calling_convention():
    p = render_prompt([{"role": "system", "content": "SYS"}, {"role": "user", "content": "x"}], TOOLS)
    assert p.startswith("<|im_start|>system\n# Tools")
    assert "<function=example_function_name>" in p
    assert '"name": "bash"' in p
    assert p.index("SYS") > p.index("</IMPORTANT>")


def test_tojson_is_not_html_escaped():
    tools = [{"type": "function", "function": {"name": "t", "description": "a <b> & 'q' ünï", "parameters": {"type": "object", "properties": {}}}}]
    p = render_prompt([{"role": "user", "content": "x"}], tools)
    assert "a <b> & 'q' ünï" in p


def test_assistant_tool_call_and_tool_response_roundtrip():
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "reasoning_content": "think", "tool_calls": [
            {"function": {"name": "bash", "arguments": {"command": "ls", "timeout": 5}}}]},
        {"role": "tool", "content": "a.txt"},
    ]
    p = render_prompt(msgs, TOOLS)
    assert "<think>\nthink\n</think>\n\n<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n<parameter=timeout>\n5\n</parameter>\n</function>\n</tool_call><|im_end|>" in p
    assert "<|im_start|>user\n<tool_response>\na.txt\n</tool_response><|im_end|>" in p


def test_reasoning_dropped_before_last_user_query():
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "reasoning_content": "OLD"},
        {"role": "user", "content": "q2"},
    ]
    assert "OLD" not in render_prompt(msgs)


def test_arguments_given_as_json_string_are_accepted():
    msgs = [{"role": "user", "content": "go"}, {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "bash", "arguments": '{"command": "pwd"}'}}]}]
    assert "<parameter=command>\npwd\n</parameter>" in render_prompt(msgs, TOOLS)


def test_complex_argument_values_are_json_encoded():
    msgs = [{"role": "user", "content": "go"}, {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "x", "arguments": {"todos": [{"content": "a", "status": "pending"}]}}}]}]
    assert '[{"content": "a", "status": "pending"}]' in render_prompt(msgs, TOOLS)


def test_no_user_query_raises():
    with pytest.raises(TemplateError):
        render_prompt([{"role": "system", "content": "S"}])


def test_matches_transformers_reference_if_available():
    """Byte-for-byte equality with `tokenizer.apply_chat_template` (ground truth)."""
    transformers = pytest.importorskip("transformers")
    import pathlib

    from app.llm.template import TEMPLATE_PATH

    tok_path = pathlib.Path(__file__).resolve().parents[1] / "app/llm/assets/tokenizer.json"
    if not tok_path.exists():
        pytest.skip("tokenizer.json not fetched")
    tok = transformers.PreTrainedTokenizerFast(tokenizer_file=str(tok_path))
    tmpl = TEMPLATE_PATH.read_text()
    convs = [
        [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}],
        [{"role": "system", "content": "S"}, {"role": "user", "content": "x"},
         {"role": "assistant", "content": "ok", "reasoning_content": "r", "tool_calls": [
             {"function": {"name": "bash", "arguments": {"command": "echo <>", "timeout": 3}}}]},
         {"role": "tool", "content": "out"}],
    ]
    for conv in convs:
        for think in (True, False):
            for tools in (None, TOOLS):
                ref = tok.apply_chat_template(conv, tools=tools, tokenize=False, add_generation_prompt=True,
                                              chat_template=tmpl, enable_thinking=think)
                assert render_prompt(conv, tools, enable_thinking=think) == ref
