from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from app.llm.chat_model import looks_degenerate, to_template_messages

CALL = "<tool_call>\n<function=add>\n<parameter=a>\n2\n</parameter>\n<parameter=b>\n3\n</parameter>\n</function>\n</tool_call>"


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


async def test_tool_call_is_typed_and_reasoning_is_separated(make_model):
    model, be = make_model(["I should add.\n</think>\n\nLet me compute." + CALL])
    ai = await model.bind_tools([add]).ainvoke([SystemMessage(content="S"), HumanMessage(content="2+3?")])
    assert ai.tool_calls[0]["name"] == "add" and ai.tool_calls[0]["args"] == {"a": 2, "b": 3}
    assert ai.content == "Let me compute."
    assert ai.additional_kwargs["reasoning_content"] == "I should add."
    # prompt really is the model-native template, in raw mode
    prompt = be.requests[0].prompt
    assert prompt.endswith("<|im_start|>assistant\n<think>\n") and "<tool_call>" in prompt
    assert "<|im_end|>" in be.requests[0].stop and "<tool_response>" in be.requests[0].stop
    assert "</thought>" not in be.requests[0].stop


async def test_streaming_emits_reasoning_content_and_calls_separately(make_model):
    model, _ = make_model(["r1 r2\n</think>\n\nhello " + CALL])
    kinds = []
    async for ch in model.bind_tools([add]).astream([HumanMessage(content="x")]):
        if ch.additional_kwargs.get("reasoning_content"):
            kinds.append("reasoning")
        if ch.content:
            kinds.append("content")
        if ch.tool_call_chunks:
            kinds.append("call")
    assert kinds[0] == "reasoning" and "content" in kinds and kinds[-1] == "call"


async def test_final_answer_without_tools(make_model):
    model, _ = make_model(["easy\n</think>\n\n5"])
    ai = await model.ainvoke([HumanMessage(content="2+3?")])
    assert ai.content == "5" and not ai.tool_calls
    assert ai.usage_metadata["output_tokens"] > 0


async def test_thinking_budget_forces_close_and_continues(make_model):
    calls = {"n": 0}

    def fn(req, n):
        calls["n"] = n
        if n == 1:
            return "word " * 400  # never closes </think>
        assert req.prompt.rstrip().endswith("</think>"), "phase 2 must close the think block"
        return "Done: 5"

    model, be = make_model(fn=fn, thinking_budget=20)
    ai = await model.ainvoke([HumanMessage(content="q")])
    assert ai.content == "Done: 5"
    assert ai.response_metadata["thinking_forced_closed"] is True
    assert len(be.requests) == 2
    assert "word word" in be.requests[1].prompt  # reasoning so far is preserved


async def test_degenerate_loop_in_thinking_is_cut_early(make_model):
    loop = "Let me check again whether the answer is five or maybe six, hmm. " * 60

    def fn(req, n):
        return loop if n == 1 else "5"

    model, _ = make_model(fn=fn, thinking_budget=1500)
    ai = await model.ainvoke([HumanMessage(content="q")])
    assert ai.response_metadata["thinking_degenerate"] is True
    assert ai.content == "5"


async def test_empty_turn_retries_in_nothink_mode(make_model):
    def fn(req, n):
        if n == 1:
            return "pondering...\n</think>\n\n"  # nothing visible
        assert req.prompt.endswith("<think>\n\n</think>\n\n")
        return "Final answer"

    model, be = make_model(fn=fn)
    ai = await model.ainvoke([HumanMessage(content="q")])
    assert ai.content == "Final answer" and ai.response_metadata["nothink_fallback"]
    assert be.requests[1].sampling.temperature <= 0.3


async def test_malformed_call_is_reported_not_crashing(make_model):
    model, _ = make_model(["x</think>\n\n<tool_call>\nthis is not valid\n</tool_call>"], empty_retries=0)
    ai = await model.bind_tools([add]).ainvoke([HumanMessage(content="q")])
    assert not ai.tool_calls
    assert ai.additional_kwargs["tool_call_errors"][0]["error"]


async def test_unknown_tool_name_is_fuzzy_resolved(make_model):
    model, _ = make_model(["x</think>\n\n" + CALL.replace("add", "Add")])
    ai = await model.bind_tools([add]).ainvoke([HumanMessage(content="q")])
    assert ai.tool_calls[0]["name"] == "add"


async def test_forced_tool_choice_prefills_header_and_disables_thinking(make_model):
    class Plan(BaseModel):
        """A plan."""
        goal: str
        steps: list[str]

    model, be = make_model(["<parameter=goal>\nShip\n</parameter>\n<parameter=steps>\n[\"a\", \"b\"]\n</parameter>\n</function>\n</tool_call>"])
    plan = await model.aextract([HumanMessage(content="plan")], Plan, name="make_plan")
    assert plan.goal == "Ship" and plan.steps == ["a", "b"]
    assert be.requests[0].prompt.endswith("<think>\n\n</think>\n\n<tool_call>\n<function=make_plan>\n")


async def test_with_structured_output_runnable(make_model):
    class Answer(BaseModel):
        """An answer."""
        value: int

    model, _ = make_model(["<parameter=value>\n42\n</parameter>\n</function>\n</tool_call>"])
    out = await model.with_structured_output(Answer).ainvoke([HumanMessage(content="q")])
    assert out.value == 42


def test_message_conversion_rules():
    msgs = [
        SystemMessage(content="A"), SystemMessage(content="B"),
        HumanMessage(content="hi"),
        AIMessage(content="", tool_calls=[{"name": "add", "args": {"a": 1, "b": 2}, "id": "1"}],
                  additional_kwargs={"reasoning_content": "r"}),
        ToolMessage(content="3", tool_call_id="1"),
        SystemMessage(content="late note"),
    ]
    out = to_template_messages(msgs)
    assert out[0] == {"role": "system", "content": "A\n\nB"}
    assert out[2]["tool_calls"][0]["function"]["arguments"] == {"a": 1, "b": 2}
    assert out[2]["reasoning_content"] == "r"
    assert out[3]["role"] == "tool"
    assert out[4]["role"] == "user" and "late note" in out[4]["content"]


def test_conversion_guarantees_a_user_query():
    out = to_template_messages([SystemMessage(content="S"), ToolMessage(content="x", tool_call_id="1")])
    assert out[-1]["role"] == "user"


def test_looks_degenerate():
    assert looks_degenerate("abc " * 200)
    assert not looks_degenerate("a perfectly normal reasoning trace " * 2)


def test_sync_invoke_works(make_model):
    model, _ = make_model(["ok\n</think>\n\nhi"])
    assert model.invoke([HumanMessage(content="x")]).content == "hi"


def test_prompt_token_counting_uses_real_tokenizer_if_present(make_model):
    model, _ = make_model([])
    n = model.count_prompt_tokens([SystemMessage(content="S"), HumanMessage(content="hello")], [add])
    assert 100 < n < 1000


async def test_usage_is_estimated_when_server_reports_none(make_model):
    from tests.conftest import ScriptedBackend
    from app.llm.backends import StreamPiece

    class NoUsage(ScriptedBackend):
        async def stream(self, req):
            yield StreamPiece(text="ok\n</think>\n\nhello")
            yield StreamPiece(done=True, finish_reason="stop")

    from app.llm.chat_model import A1ChatModel
    from app.llm.profile import SamplingParams
    model = A1ChatModel(backend=NoUsage(), sampling=SamplingParams())
    ai = await model.ainvoke([HumanMessage(content="hi")])
    assert ai.usage_metadata["input_tokens"] > 5 and ai.usage_metadata["output_tokens"] > 0
