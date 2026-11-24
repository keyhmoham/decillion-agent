import type { ApiMessage, ServerEvent, TeamTask, TimelineItem, Todo } from "../api/types";

export interface ChatState {
  items: TimelineItem[];
  todos: Todo[];
  tasks: TeamTask[];
  running: boolean;
  runId: string | null;
  status: string;            // transient "what is it doing" label
  usage: Record<string, number>;
  context: { tokens: number; budget: number };
  mode?: string;
  modeReason?: string;
  dirtyFiles: number;        // bumps when the agent changed files
  memoryTick: number;        // bumps when memory changed
  lastSeq: number;
  error?: string;
}

export const emptyChat = (): ChatState => ({
  items: [], todos: [], tasks: [], running: false, runId: null, status: "", usage: {}, context: { tokens: 0, budget: 0 },
  dirtyFiles: 0, memoryTick: 0, lastSeq: 0,
});

/** Rebuild the timeline from persisted rows (page load). */
export function itemsFromMessages(msgs: ApiMessage[], skipRunId?: string | null): TimelineItem[] {
  const items: TimelineItem[] = [];
  const toolIndex = new Map<string, number>();
  for (const m of msgs) {
    if (skipRunId && m.run_id === skipRunId && m.role !== "user") continue;
    if (m.role === "user") items.push({ kind: "user", id: `m${m.id}`, content: m.content });
    else if (m.role === "assistant") {
      items.push({ kind: "assistant", id: `m${m.id}`, agent: m.agent, taskKey: m.meta?.task_key, reasoning: m.reasoning, content: m.content, toolCallIds: m.tool_calls.map((t) => t.id), streaming: false, flags: m.meta?.flags });
      for (const t of m.tool_calls) {
        toolIndex.set(t.id, items.length);
        items.push({ kind: "tool", id: t.id, name: t.name, args: t.args, agent: m.agent, output: "", status: "running" });
      }
    } else if (m.role === "tool" && m.tool_call_id) {
      const idx = toolIndex.get(m.tool_call_id);
      if (idx !== undefined) {
        const it = items[idx];
        if (it.kind === "tool") Object.assign(it, { result: m.content, ok: m.meta?.ok !== false, status: m.meta?.ok === false ? "error" : "done", duration: m.meta?.duration });
      }
    } else if (m.role === "card") items.push({ kind: "card", id: `m${m.id}`, agent: m.agent, card: m.meta.card });
    else if (m.role === "notice") items.push({ kind: "notice", id: `m${m.id}`, level: m.meta?.level ?? "info", text: m.content });
  }
  // Items of a still-live run are skipped above, so any call without a result belongs to a finished (cancelled/crashed) run.
  for (const it of items) if (it.kind === "tool" && it.status === "running" && it.result === undefined) it.status = "error";
  return items;
}

function updateAssistant(items: TimelineItem[], id: string, fn: (a: Extract<TimelineItem, { kind: "assistant" }>) => void) {
  const it = items.find((x) => x.kind === "assistant" && x.id === id);
  if (it && it.kind === "assistant") fn(it);
}

/** Pure-ish reducer: mutates a cloned state. */
export function applyEvent(prev: ChatState, e: ServerEvent): ChatState {
  const s: ChatState = { ...prev, items: [...prev.items] };
  if (e.seq) s.lastSeq = Math.max(s.lastSeq, e.seq);
  const agent = e.agent ?? "main";
  switch (e.type) {
    case "hello":
      s.running = !!e.running;
      break;
    case "user_message":
      if (!s.items.some((i) => i.kind === "user" && i.id === `m${e.message?.id}`)) s.items.push({ kind: "user", id: `m${e.message?.id ?? Math.random()}`, content: e.message?.content ?? "" });
      break;
    case "run_started":
      s.running = true; s.runId = e.run_id ?? null; s.error = undefined; s.status = "Starting…"; s.tasks = [];
      break;
    case "status":
      s.status = e.text ?? "";
      break;
    case "mode_selected":
      s.mode = e.mode; s.modeReason = e.reason;
      break;
    case "message_start":
      s.items.push({ kind: "assistant", id: e.message_id, agent, taskKey: e.task_key, reasoning: "", content: "", toolCallIds: [], streaming: true });
      s.status = "Thinking…";
      break;
    case "reasoning_delta":
      updateAssistant(s.items, e.message_id, (a) => { a.reasoning += e.text; });
      s.items = [...s.items];
      break;
    case "content_delta":
      updateAssistant(s.items, e.message_id, (a) => { a.content += e.text; });
      s.items = [...s.items];
      s.status = "Writing…";
      break;
    case "message_end": {
      let found = false;
      updateAssistant(s.items, e.message_id, (a) => {
        found = true;
        a.streaming = false; a.content = e.content ?? a.content; a.reasoning = e.reasoning ?? a.reasoning; a.flags = e.flags;
        a.toolCallIds = (e.tool_calls ?? []).map((t: any) => t.id);
      });
      if (!found) s.items.push({ kind: "assistant", id: e.message_id, agent, taskKey: e.task_key, reasoning: e.reasoning ?? "", content: e.content ?? "", toolCallIds: [], streaming: false });
      s.items = [...s.items];
      break;
    }
    case "message_error":
      updateAssistant(s.items, e.message_id, (a) => { a.streaming = false; });
      break;
    case "tool_start":
      s.items.push({ kind: "tool", id: e.call_id, name: e.name, args: e.args ?? {}, agent, output: "", status: "running" });
      s.status = `Running ${e.name}…`;
      break;
    case "tool_output": {
      const t = s.items.find((x) => x.kind === "tool" && x.id === e.call_id);
      if (t && t.kind === "tool") { t.output = (t.output + e.text).slice(-60000); s.items = [...s.items]; }
      break;
    }
    case "tool_end": {
      const t = s.items.find((x) => x.kind === "tool" && x.id === e.call_id);
      if (t && t.kind === "tool") { t.result = e.result; t.ok = e.ok; t.status = e.ok ? "done" : "error"; t.duration = e.duration; s.items = [...s.items]; }
      break;
    }
    case "card":
      s.items.push({ kind: "card", id: e.card.id, agent, card: e.card });
      break;
    case "notice":
      s.items.push({ kind: "notice", id: `n${e.seq}`, level: e.level ?? "info", text: e.text });
      break;
    case "approval_request":
      s.items.push({ kind: "approval", id: `a${e.approval_id}`, approvalId: e.approval_id, tool: e.tool, args: e.args, status: "pending" });
      s.status = "Waiting for your approval";
      break;
    case "approval_resolved":
      s.items = s.items.map((i) => (i.kind === "approval" && i.approvalId === e.approval_id ? { ...i, status: e.approved ? "approved" : "denied" } : i));
      break;
    case "todos":
      if (agent === "main") s.todos = e.todos;
      break;
    case "plan":
      s.tasks = e.tasks;
      s.items.push({ kind: "plan", id: `plan${e.seq}`, summary: e.summary, tasks: e.tasks });
      break;
    case "tasks":
      s.tasks = e.tasks;
      s.items = s.items.map((i) => (i.kind === "plan" ? { ...i, tasks: e.tasks } : i));
      break;
    case "usage":
      s.usage = e.usage;
      break;
    case "context":
      if (agent === "main") s.context = { tokens: e.tokens, budget: e.budget };
      break;
    case "file_changed":
      s.dirtyFiles++;
      break;
    case "memory": case "core_memory":
      s.memoryTick++;
      break;
    case "error":
      s.error = e.message;
      s.items.push({ kind: "notice", id: `err${e.seq ?? Math.random()}`, level: "error", text: e.message });
      break;
    case "run_finished":
      s.running = false; s.status = ""; s.usage = e.usage ?? s.usage;
      s.items = s.items.map((i) => (i.kind === "tool" && i.status === "running" ? { ...i, status: "error" as const } : i.kind === "assistant" && i.streaming ? { ...i, streaming: false } : i));
      s.dirtyFiles++;
      break;
    default:
      break;
  }
  return s;
}
