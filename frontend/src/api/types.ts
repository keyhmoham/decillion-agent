export type Mode = "auto" | "agent" | "team";

export interface SessionConfig {
  mode: Mode;
  thinking: boolean;
  approval: "auto" | "ask";
  mcp: string[];
  custom_mcp: { name: string; command?: string; url?: string }[];
  team_parallelism: number;
  memory: boolean;
}

export interface Todo { content: string; status: "pending" | "in_progress" | "completed" }

export interface Session {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  config: SessionConfig;
  todos: Todo[];
  core_memory: Record<string, string>;
  summary: string;
  usage: Record<string, number>;
  running?: boolean;
}

export interface ToolCallRef { id: string; name: string; args: Record<string, unknown> }

export interface ApiMessage {
  id: number;
  run_id: string | null;
  role: "user" | "assistant" | "tool" | "card" | "notice";
  agent: string;
  content: string;
  reasoning: string;
  tool_calls: ToolCallRef[];
  tool_call_id: string | null;
  tool_name: string | null;
  meta: Record<string, any>;
  created_at: string;
}

export interface Card { id: string; kind: "table" | "chart" | "stats" | "diagram" | "html" | "image" | "callout"; title: string; data: any }

export interface TeamTask {
  key: string; title: string; description: string; role: string;
  status: "pending" | "running" | "done" | "failed" | "blocked";
  depends_on: string[]; result: string; attempts: number; handoff_from?: string | null;
}

export interface McpPreset {
  id: string; name: string; description: string; category: string; default: boolean;
  approx_tools: number; approx_prompt_tokens: number; available: boolean; requires: string[];
}

export interface AppConfig {
  version: string;
  model: { tag: string; family: string; parameters: string; quantization: string; hf_repo: string; backend: string; num_ctx: number; context_budget: number; thinking_budget: number };
  defaults: { mode: Mode; thinking: boolean; max_steps: number };
  sandbox: { backend: string; image: string; memory: string; cpus: number; network: string };
  mcp_catalog: McpPreset[];
  roles: { name: string; title: string; description: string; color: string }[];
  auth_required: boolean;
}

export interface FileEntry { name: string; path: string; is_dir: boolean; size: number; mtime: number }
export interface MemoryItem { id: string; session_id: string | null; kind: string; content: string; importance: number; tags: string[]; source: string; updated_at: string | null }

export interface ServerEvent { seq?: number; type: string; run_id?: string; agent?: string; [k: string]: any }

// ---- timeline model rendered by the chat -------------------------------------------------
export type TimelineItem =
  | { kind: "user"; id: string; content: string }
  | { kind: "assistant"; id: string; agent: string; taskKey?: string | null; reasoning: string; content: string; toolCallIds: string[]; streaming: boolean; flags?: Record<string, any> }
  | { kind: "tool"; id: string; name: string; args: Record<string, unknown>; agent: string; output: string; result?: string; ok?: boolean; status: "running" | "done" | "error" | "awaiting"; duration?: number }
  | { kind: "card"; id: string; agent: string; card: Card }
  | { kind: "notice"; id: string; level: string; text: string }
  | { kind: "approval"; id: string; approvalId: string; tool: string; args: Record<string, unknown>; status: "pending" | "approved" | "denied" }
  | { kind: "plan"; id: string; summary: string; tasks: TeamTask[] };
