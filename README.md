# Decillion Agent

A production-style **agentic system built specifically for `nutboy02/Agents-A1-4B-Kimi-heretic`**
(a 4B Qwen3.5-architecture agent model): LangGraph agents with tool calling, thinking, planning,
sub-agent delegation, an orchestrated **agent team** with task dispatch, MCP servers, layered
memory, **one Docker sandbox computer per chat session**, and a Claude-Code/Codex-style React UI
that renders **charts, tables, KPI tiles, diagrams and other visual cards** produced by the agents.

```bash
docker compose up --build        # then open http://localhost:3000
```

First start pulls the model (~3.3 GB) into the `ollama` volume and builds the sandbox image
(a few minutes). GPU: `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build`.
Everything has a default; copy `.env.example` to `.env` only to change something.

| Service | Role |
|---|---|
| `frontend` (nginx + React/Vite/Ant Design) | UI, reverse proxy for `/api` + WebSocket |
| `backend` (FastAPI, LangChain/LangGraph) | agents, tools, MCP, memory, sandbox manager |
| `postgres` | sessions, messages, tasks, long-term memory (FTS + trigram), LangGraph checkpoints |
| `ollama` + `model-init` | serves the model; pulls it on first start |
| `sandbox-image` | builds the per-session agent computer, then exits |

![Agent run with tool cards](docs/screenshots/agent-run.png)
![Charts, tables and diagrams rendered as cards](docs/screenshots/visual-cards.png)
![Team mode: task board and per-agent streams](docs/screenshots/team-mode.png)

## What makes it specific to this model

The Ollama package's `TEMPLATE` is **not** the format the model was trained on (it uses `<thought>`
tags, drops tool calls/results from history, adds a wrong stop token). So this system renders the
model's real chat template itself, talks to the server in raw mode, and parses the model's native
`<think>` / `<tool_call><function=…><parameter=…>` stream. Details, measurements and design
responses are in [docs/MODEL_NOTES.md](docs/MODEL_NOTES.md). Highlights:

* byte-exact prompt rendering (verified against `transformers`), incremental stream parser, typed
  argument coercion, malformed-call recovery with actionable error feedback;
* thinking budget + repetition guard, empty-turn fallback to the model's no-think mode;
* forced-schema structured output (pre-filled `<function=…>`) for planner/reviewer/triage/memory;
* context compaction tuned to a model fine-tuned on short traces; KV-cache-friendly prompts.

## Agent system

```
                       ┌───────────────  AgentService (per chat turn) ────────────────┐
 user ── WebSocket ──► │ recall memories → connect MCP → triage(auto) → graph → reflect │ ──► events ► UI
                       └──────────────┬─────────────────────────┬───────────────────────┘
                          mode=agent  │                         │ mode=team
                 ┌────────────────────▼───────┐      ┌──────────▼───────────────────────────────┐
                 │ ReAct graph (LangGraph)    │      │ plan(think→structured) → dispatch (Send, │
                 │ agent ⇄ tools              │      │ parallel, dependency-aware) → worker     │
                 │ format_fix / empty_fix     │◄─────│ (same graph, role tools) → review/retry  │
                 │ finish_check / finalize    │      │ → hand-offs → synthesize                 │
                 └────────────────────────────┘      └──────────────────────────────────────────┘
```

* **Tools**: `bash` (streaming output, background jobs), `read_file`/`write_file`/`edit_file`,
  `list_dir`/`glob`/`grep`, `web_search`/`web_fetch` (run *inside* the sandbox, SSRF-guarded),
  `write_todos`/`update_todo`, memory tools, `delegate_task`, `handoff`, and presentation tools
  `show_table`, `show_chart`, `show_stats`, `show_diagram` (Mermaid), `show_html` (isolated iframe),
  `show_image`, `show_callout`.
* **Hardening**: JSON-schema validation before execution, repeated-call detection, plan completion
  push-back, step budget with forced summary, dangling-call repair after cancellation, per-tool
  timeouts, optional human approval for mutating tools.
* **Team**: roles `researcher`, `coder`, `tester`, `reviewer`, `writer`, `devops`; the orchestrator
  plans (free thinking → strict structured extraction), validates/repairs the DAG, dispatches ready
  tasks in parallel, runs an acceptance review with feedback-driven retry, supports swarm-style
  hand-offs, blocks dependents of failed tasks and synthesises the final report.
* **Memory**: (1) *core blocks* always in context (`user`/`project`/`notes`, MemGPT-style, editable
  by agent and user), (2) *archival memory* in Postgres with FTS + trigram + importance + recency
  ranking, de-duplication and scopes (global / session), (3) conversation compaction, (4) automatic
  post-run reflection that saves durable lessons.
* **MCP** (run inside the session sandbox over `docker exec -i`): time, git, sequential-thinking
  (default on), fetch, filesystem, knowledge-graph memory, Playwright, Context7, SQLite, GitHub,
  Brave Search. The UI shows the prompt-token cost per server because a 4B model prefers a short tool list.
* **Sandbox**: one hardened container + workspace volume per session (cap-drop ALL, no-new-privileges,
  memory/CPU/PID limits, non-root user, configurable network), created lazily, stopped when idle,
  removed with the session. Files, terminal and workspace download are exposed in the UI.

## Verification status (honest summary)

| Layer | How it was verified |
|---|---|
| Prompt/template | byte-equal to `transformers.apply_chat_template` (pytest) |
| Parser / tool-call parsing | exhaustive chunk-boundary fuzz + malformed-input tests |
| **Model compatibility** | `scripts/verify_model.py`: all 6 live checks pass against the real IQ4_NL GGUF (served by llama.cpp via llama-cpp-python): direct answer, tool call, typed array/int arguments, tool-result follow-up, structured extraction, think/no-think |
| **Full agentic run, real model** | end to end through API + PostgreSQL: the model planned, wrote a file, executed it and answered correctly; it also emitted valid `show_chart` / `show_table` calls (nested series/rows) and a valid 4-task team plan via think-then-structured-extraction |
| Graph / team / memory / API | 99 backend tests: scripted model + real PostgreSQL (LangGraph Postgres checkpointer, WebSocket, memory FTS/trigram), real MCP server over stdio |
| Frontend | `tsc`, production build, 6 vitest reducer tests, Playwright runs against the real API (agent, team, approvals, auto-scroll, zero console errors) |
| **Not run in the build environment** | the `docker compose up` stack itself, the Ollama serving path, and the Docker sandbox against a real daemon (no Docker daemon / GitHub-hosted binaries were available). Compose files pass `docker compose config` (incl. GPU/llama.cpp overrides), the backend dependency set resolves, and the Docker sandbox logic is tested against a faithful fake of the docker SDK; the tool layer was exercised through the equivalent local sandbox. Run the checklist below on first deployment. |

First-deployment checklist: `docker compose up --build` → open the UI → *Context → Run self-test*
(all green) → ask for the "Build a CLI app" example → confirm a `a1-sbx-*` container appears
(`docker ps`) and disappears when the session is deleted.

## Development

```bash
# backend tests (PostgreSQL needed for the DB/API tests)
cd backend && pip install -e . pytest pytest-asyncio transformers
TEST_DATABASE_URL=postgresql+asyncpg://agent:agent@localhost:5432/agent_test pytest -q
# frontend
cd frontend && npm ci && npm run dev          # proxies /api to http://127.0.0.1:8000
# UI development without a model: scripted agent that produces every card type
cd backend && TEST_DATABASE_URL=... python -m tests.demo_server --port 8010
```

`SANDBOX_BACKEND=local` runs tools directly on the backend host (tests/dev only - **not isolated**).
More: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/SECURITY.md](docs/SECURITY.md).

## Configuration

See [.env.example](.env.example). The most relevant knobs for this model: `LLM_NUM_CTX`,
`THINKING_BUDGET`, `CONTEXT_BUDGET`, `TEAM_PARALLELISM`, `SANDBOX_*`, `APP_API_KEY`.
Alternative model servers: `docker-compose.llamacpp.yml` (experimental) or any server exposing
`/v1/completions` (`LLM_BACKEND=openai`).

## License

Application code: MIT. The model weights have their own licences (fine-tune/GGUF: AGPL-3.0; base
Agents-A1-4B: Apache-2.0); the model is *uncensored* - review the disclaimers on the model cards.
