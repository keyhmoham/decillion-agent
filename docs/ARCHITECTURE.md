# Architecture

```
backend/app
├── llm/            model layer (template, raw backends, stream parser, tool-call parser, LangChain ChatModel, self-test)
├── agents/         graph.py (ReAct), team.py (orchestration), workers.py (role agents), service.py (turn runner),
│                   compaction.py, events.py (bus + recorder), prompts.py, roles.py
├── tools/          sandbox_tools, web_tools(+web_impl), plan_tools, memory_tools, team_tools, card_tools, registry
├── sandbox/        base interface, docker_sandbox (production), local_sandbox (tests/dev), manager (lifecycle/reaper)
├── mcp/            catalog of presets + manager (per-session persistent connections, isolated failures)
├── memory/         Postgres long-term memory store
├── db/             SQLAlchemy models, repository, session/bootstrap
└── api/            REST + WebSocket routes, auth dependency
frontend/src        store (zustand) · event reducer · components (chat, cards, panels)
```

## One chat turn

1. `POST /messages` or WS `{"type":"send"}` → `AgentService.start_run` persists the user message and
   spawns an asyncio task (one active run per session).
2. The task ensures the session sandbox, connects the selected MCP servers *inside* it, recalls
   relevant memories, and (mode `auto`) triages solo vs team (cheap heuristics first, then a
   no-think structured classification).
3. The LangGraph graph runs; every LLM call goes through `A1ChatModel` (raw prompt → stream parser →
   typed tool calls). `RunCallbacks` turns token streams into `reasoning_delta`/`content_delta`
   events and persists finished messages; tool nodes emit `tool_start`/`tool_output`/`tool_end`.
4. Conversation state is checkpointed per session in Postgres (LangGraph `AsyncPostgresSaver`);
   the UI transcript (messages, cards, notices) is stored separately for rich replay.
5. After the run: reflection (durable memories), title, usage, `run_finished`.

The event bus keeps a replay buffer, so a reloaded browser tab resumes a live run (`?since=seq`).

## Failure model

* model server down → run fails with a clear event, session intact;
* malformed / unknown / invalid tool calls → corrective tool response, bounded retries;
* stuck loops → warning, then forced summary; step budget → forced summary;
* cancelled run → dangling tool calls repaired on the next turn;
* backend restart → interrupted runs are marked failed at start-up;
* MCP server crash/slow start → isolated, reported once, run continues.
