import { describe, expect, it } from "vitest";
import type { ApiMessage, ServerEvent } from "../api/types";
import { applyEvent, emptyChat, itemsFromMessages } from "./reducer";

const run = (events: ServerEvent[]) => events.reduce(applyEvent, emptyChat());

describe("applyEvent", () => {
  it("streams reasoning and content into one assistant item, then finalises it", () => {
    const s = run([
      { type: "run_started", run_id: "r1" },
      { type: "message_start", message_id: "m1", agent: "main" },
      { type: "reasoning_delta", message_id: "m1", text: "think " },
      { type: "reasoning_delta", message_id: "m1", text: "hard" },
      { type: "content_delta", message_id: "m1", text: "Hel" },
      { type: "content_delta", message_id: "m1", text: "lo" },
      { type: "message_end", message_id: "m1", content: "Hello", reasoning: "think hard", tool_calls: [], flags: { thinking_forced_closed: true } },
    ]);
    const a = s.items[0];
    expect(a.kind === "assistant" && a.content).toBe("Hello");
    expect(a.kind === "assistant" && a.reasoning).toBe("think hard");
    expect(a.kind === "assistant" && a.streaming).toBe(false);
    expect(a.kind === "assistant" && a.flags?.thinking_forced_closed).toBe(true);
    expect(s.running).toBe(true);
  });

  it("tracks a tool call from start through streamed output to result", () => {
    const s = run([
      { type: "tool_start", call_id: "c1", name: "bash", args: { command: "ls" }, agent: "main" },
      { type: "tool_output", call_id: "c1", stream: "stdout", text: "a\n" },
      { type: "tool_output", call_id: "c1", stream: "stdout", text: "b\n" },
      { type: "tool_end", call_id: "c1", name: "bash", result: "a\nb\n[exit code 0]", ok: true, duration: 0.2 },
    ]);
    const t = s.items[0];
    expect(t.kind).toBe("tool");
    if (t.kind === "tool") {
      expect(t.output).toBe("a\nb\n");
      expect(t.status).toBe("done");
      expect(t.result).toContain("exit code 0");
    }
  });

  it("renders cards, plans, todos (main agent only) and approvals", () => {
    const s = run([
      { type: "card", agent: "main", card: { id: "k1", kind: "chart", title: "T", data: {} } },
      { type: "todos", agent: "main", todos: [{ content: "a", status: "pending" }] },
      { type: "todos", agent: "coder:t1", todos: [{ content: "worker-only", status: "pending" }] },
      { type: "plan", seq: 5, summary: "s", tasks: [{ key: "t1", title: "x", role: "coder", status: "pending", depends_on: [], description: "", result: "", attempts: 0 }] },
      { type: "tasks", tasks: [{ key: "t1", title: "x", role: "coder", status: "done", depends_on: [], description: "", result: "", attempts: 1 }] },
      { type: "approval_request", approval_id: "a1", tool: "bash", args: { command: "rm x" } },
      { type: "approval_resolved", approval_id: "a1", approved: false },
    ]);
    expect(s.items.map((i) => i.kind)).toEqual(["card", "plan", "approval"]);
    expect(s.todos).toHaveLength(1);
    const plan = s.items[1];
    expect(plan.kind === "plan" && plan.tasks[0].status).toBe("done");
    const ap = s.items[2];
    expect(ap.kind === "approval" && ap.status).toBe("denied");
  });

  it("run_finished stops spinners on unfinished tools and streaming messages", () => {
    const s = run([
      { type: "message_start", message_id: "m", agent: "main" },
      { type: "tool_start", call_id: "c", name: "bash", args: {}, agent: "main" },
      { type: "run_finished", status: "cancelled", usage: { llm_calls: 1 } },
    ]);
    expect(s.running).toBe(false);
    expect(s.items.some((i) => i.kind === "tool" && i.status === "running")).toBe(false);
    expect(s.items.some((i) => i.kind === "assistant" && i.streaming)).toBe(false);
  });
});

describe("itemsFromMessages", () => {
  const base = { run_id: "r", agent: "main", reasoning: "", tool_calls: [], tool_call_id: null, tool_name: null, meta: {}, created_at: "" };
  const msgs: ApiMessage[] = [
    { ...base, id: 1, role: "user", content: "hi" },
    { ...base, id: 2, role: "assistant", content: "", tool_calls: [{ id: "c1", name: "bash", args: { command: "ls" } }], reasoning: "r" },
    { ...base, id: 3, role: "tool", content: "out", tool_call_id: "c1", tool_name: "bash", meta: { ok: true, duration: 1 } },
    { ...base, id: 4, role: "card", content: "", meta: { card: { id: "k", kind: "table", title: "T", data: {} } } },
    { ...base, id: 5, role: "assistant", content: "done" },
  ];
  it("joins tool results to their calls and keeps order", () => {
    const items = itemsFromMessages(msgs);
    expect(items.map((i) => i.kind)).toEqual(["user", "assistant", "tool", "card", "assistant"]);
    const t = items[2];
    expect(t.kind === "tool" && t.status === "done" && t.result === "out").toBe(true);
  });
  it("marks calls without a result as errors (crashed run) unless that run is still live", () => {
    const dangling = msgs.filter((m) => m.role !== "tool");
    expect((itemsFromMessages(dangling)[2] as any).status).toBe("error");
    expect((itemsFromMessages(dangling, "other")[2] as any).status).toBe("error");
    expect(itemsFromMessages(dangling, "r").some((i) => i.kind === "assistant")).toBe(false); // live run replayed from events
  });
});
