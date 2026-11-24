import type { ApiMessage, AppConfig, FileEntry, MemoryItem, Session, SessionConfig, TeamTask, Todo } from "./types";

let apiKey = localStorage.getItem("decillion.apiKey") ?? "";
export const getApiKey = () => apiKey;
export function setApiKey(k: string) {
  apiKey = k;
  localStorage.setItem("decillion.apiKey", k);
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function req<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = { ...(init.headers as Record<string, string>) };
  if (apiKey) headers["x-api-key"] = apiKey;
  if (init.body && !(init.body instanceof FormData)) headers["Content-Type"] = "application/json";
  const res = await fetch(`/api${path}`, { ...init, headers });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = j.detail ?? detail;
    } catch { /* not json */ }
    throw new ApiError(res.status, typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

const q = (o: Record<string, string | number | boolean | undefined>) =>
  "?" + Object.entries(o).filter(([, v]) => v !== undefined).map(([k, v]) => `${k}=${encodeURIComponent(String(v))}`).join("&");

export interface SessionState {
  running: boolean; run_id: string | null; todos: Todo[]; core_memory: Record<string, string>; usage: Record<string, number>;
  summary: string; tasks: TeamTask[]; sandbox: Record<string, any>; mcp: { name: string; tools: string[]; error: string | null }[]; last_seq: number;
}

export const api = {
  health: () => req<{ status: string; model: any; database: boolean; auth_required: boolean }>("/health"),
  config: () => req<AppConfig>("/config"),
  selftest: () => req<{ passed: boolean; results: { name: string; passed: boolean; detail: string; seconds: number }[] }>("/model/selftest", { method: "POST" }),
  sessions: () => req<Session[]>("/sessions"),
  createSession: (config?: Partial<SessionConfig>) => req<Session>("/sessions", { method: "POST", body: JSON.stringify({ config: config ?? {} }) }),
  session: (id: string) => req<Session>(`/sessions/${id}`),
  patchSession: (id: string, body: { title?: string; config?: Partial<SessionConfig> }) => req<Session>(`/sessions/${id}`, { method: "PATCH", body: JSON.stringify(body) }),
  deleteSession: (id: string) => req<void>(`/sessions/${id}`, { method: "DELETE" }),
  messages: (id: string) => req<ApiMessage[]>(`/sessions/${id}/messages`),
  state: (id: string) => req<SessionState>(`/sessions/${id}/state`),
  send: (id: string, content: string, options?: Partial<SessionConfig>) => req<{ run_id: string }>(`/sessions/${id}/messages`, { method: "POST", body: JSON.stringify({ content, options: options ?? {} }) }),
  cancel: (id: string) => req<{ cancelled: boolean }>(`/sessions/${id}/cancel`, { method: "POST" }),
  approve: (id: string, aid: string, approve: boolean) => req(`/sessions/${id}/approvals/${aid}`, { method: "POST", body: JSON.stringify({ approve }) }),
  files: (id: string, path = ".") => req<{ path: string; workdir: string; entries: FileEntry[] }>(`/sessions/${id}/files${q({ path })}`),
  file: (id: string, path: string) => req<{ path: string; binary: boolean; size: number; mime: string; content: string }>(`/sessions/${id}/file${q({ path })}`),
  writeFile: (id: string, path: string, content: string) => req(`/sessions/${id}/file`, { method: "PUT", body: JSON.stringify({ path, content }) }),
  upload: (id: string, files: File[], dir = ".") => {
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f));
    return req<{ saved: string[] }>(`/sessions/${id}/upload${q({ dir })}`, { method: "POST", body: fd });
  },
  terminal: (id: string, command: string, timeout = 60) => req<{ exit_code: number; stdout: string; stderr: string; timed_out: boolean; duration: number }>(`/sessions/${id}/terminal`, { method: "POST", body: JSON.stringify({ command, timeout }) }),
  restartSandbox: (id: string) => req(`/sessions/${id}/sandbox/restart`, { method: "POST" }),
  memories: (sessionId?: string) => req<MemoryItem[]>(`/memories${q({ session_id: sessionId })}`),
  addMemory: (content: string, kind: string, scope: "global" | "session", sessionId?: string) =>
    req<MemoryItem>("/memories", { method: "POST", body: JSON.stringify({ content, kind, scope, session_id: sessionId }) }),
  deleteMemory: (id: string) => req<void>(`/memories/${id}`, { method: "DELETE" }),
  setCore: (id: string, block: string, content: string) => req(`/sessions/${id}/core-memory/${block}`, { method: "PUT", body: JSON.stringify({ content }) }),
};

export const fileUrl = (sid: string, path: string) => `/api/sessions/${sid}/file${q({ path, raw: true, token: apiKey || undefined })}`;
export const downloadUrl = (sid: string) => `/api/sessions/${sid}/download${q({ path: ".", token: apiKey || undefined })}`;
export const wsUrl = (sid: string, since: number) => {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}/api/sessions/${sid}/ws${q({ since, token: apiKey || undefined })}`;
};
