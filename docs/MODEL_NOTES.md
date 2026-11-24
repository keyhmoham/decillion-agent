# Agents-A1-4B-Kimi-heretic: what we found and how the system adapts to it

This system is built *for one model*. Everything below was verified against the live model
(IQ4_NL GGUF) rather than assumed.

## Where the model comes from

| Place | What it is |
|---|---|
| Ollama `nutboy02/Agents-A1-4B-Kimi-heretic` (`:latest`, `:32K`, `:64K`) | 3.3 GB package (IQ4_NL weights 2.6 GB + vision projector). Capabilities shown: vision, tools, thinking. Its README only links to the GGUF repo below. |
| Hugging Face `hotdogs/Agents-A1-4B-kimi-Preview-heretic-GGUF` | The GGUF source of the Ollama package (IQ4_NL / Q8_0 / F16). AGPL-3.0. |
| Hugging Face `hotdogs/Agents-A1-4B-kimi-Preview-heretic` | The merged fine-tune: *Agents-A1-4B* SFT'd on `greghavens/kimi-k3-coding-and-debugging-traces` (3,389 traces, LoRA r=64, merge scale 0.4) and abliterated with *heretic* (refusals ~37%). Trained at **4,096-token** sequences. |
| Hugging Face `InternScience/Agents-A1-4B` | The base: Qwen3.5-architecture hybrid (linear + full attention) dense 4B agent model, Apache-2.0. Strong long-horizon search / tool-calling scores. |

> `nutboy02/Agents-A1-4B-Kimi-heretic` does **not** exist as a Hugging Face repository (the API
> returns 401/404); the HF resources are the `hotdogs/*` repos above.

## The key finding: Ollama's TEMPLATE is not the model's trained format

The Ollama package ships this `TEMPLATE` (abridged) and stop list:

```
{{- if .Tools }}{{ range .Tools }}{{ . }}{{ end }}{{ end }}
<|im_start|>assistant{{ if .Thinking }}<thought> ... </thought>{{ end }}
stop: <|im_start|>, <|im_end|>, </thought>
```

The model itself (GGUF metadata, HF `chat_template.jinja`, identical in base/fine-tune/GGUF) is trained on:

* reasoning inside **`<think>` ... `</think>`** (the generation prompt ends with `<think>\n`),
* a `# Tools` preamble listing tools as JSON, plus the calling convention
  `<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>`,
* tool results delivered as `<tool_response>` blocks inside a *user* turn,
* reasoning replay only for assistant turns after the last real user message.

Consequences of using the Ollama template as-is: tools are shown as bare JSON with no calling
instructions, previous tool calls/results are *not rendered at all* (multi-step agent loops break),
`<thought>` never matches what the model produces, and the extra `</thought>` stop is wrong.

**What we do instead** (`backend/app/llm/`):

1. `template.py` renders the vendored upstream Jinja template ourselves
   (verified **byte-identical** to `transformers.apply_chat_template` for thinking on/off, with
   and without tools, multi-tool history, HTML-special characters).
2. `backends.py` sends the finished prompt in **raw mode** (`/api/generate` with `raw: true` for
   Ollama; `/v1/completions` for llama.cpp / vLLM / llama-cpp-python), with stops
   `<|im_end|>`, `<|im_start|>`, `<tool_response>` only.
3. `parser.py` is an incremental state machine for `<think>` / content / `<tool_call>`, robust to
   tags split across chunks, tool calls inside the reasoning, hallucinated tool responses and
   truncated calls.
4. `toolcall.py` parses the XML calls, recovers JSON-style calls and missing closing tags,
   coerces string parameters to the tool's JSON-schema types (arrays/objects/ints/bools - the model
   emits JSON for those, verified live) and fuzzy-matches near-miss tool names.

## Behaviours we measured and designed around

| Observation (live model) | Design response |
|---|---|
| Native format works with up to ~25 tools; short imperative system prompts work best | ~430-token system prompt, numbered rules, compact tool schemas (`title`/`additionalProperties` stripped) |
| Tool call may follow text with no newline (`…file:<tool_call>`) | parser handles marker anywhere |
| No-think mode (`<think>\n\n</think>\n\n` prefilled) gives clean one-shot calls | structured extraction (planner, reviewer, triage, memory reflection) uses no-think + a pre-filled `<function=NAME>` header so output is guaranteed schema-shaped |
| Reasoning can ramble / loop | **thinking budget** (default 1800 tokens) and a repetition detector: the `<think>` block is closed for the model and generation continues |
| A turn can end empty after `</think>` | automatic retry in no-think mode at lower temperature |
| Fine-tuned at 4K tokens, long contexts degrade | history compaction at 28K prompt tokens (micro-trim of old tool output first, then model-written structured summary) |
| Prompt eval dominates on CPU | system prompt + tool block are byte-stable per session so llama.cpp/Ollama reuse the KV prefix; harness nudges are appended as tool responses (a real user message would change the replay window) |
| Published sampling | temperature 0.6, top-p 0.9, top-k 35, min-p 0.05, repeat-penalty 1.05 (from the Ollama params layer) |

## Reproducing the verification

```bash
python scripts/verify_model.py                       # 6 live checks
python -m pytest backend/tests -m realmodel          # same checks under pytest (A1_TEST_BASE_URL=...)
```

Checks: direct answer without tools; single tool call; typed array/int arguments; tool-result
follow-up; structured extraction; thinking toggle.
