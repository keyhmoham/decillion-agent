import { create } from "zustand";
import { api } from "../api/client";
import type { AppConfig, Session, SessionConfig, ServerEvent } from "../api/types";
import { applyEvent, ChatState, emptyChat, itemsFromMessages } from "../lib/reducer";
import { SessionSocket } from "../lib/ws";

export type PanelTab = "files" | "terminal" | "plan" | "memory" | "tools" | "context";

interface Store {
  config?: AppConfig;
  health?: { status: string; model: any; database: boolean };
  connected: boolean;
  sessions: Session[];
  activeId?: string;
  chats: Record<string, ChatState>;
  panelOpen: boolean;
  panelTab: PanelTab;
  booting: boolean;
  fatal?: string;

  boot: () => Promise<void>;
  refreshSessions: () => Promise<void>;
  select: (id: string) => Promise<void>;
  newSession: (config?: Partial<SessionConfig>) => Promise<void>;
  removeSession: (id: string) => Promise<void>;
  rename: (id: string, title: string) => Promise<void>;
  send: (text: string) => Promise<void>;
  cancel: () => Promise<void>;
  approve: (approvalId: string, approve: boolean) => Promise<void>;
  patchConfig: (patch: Partial<SessionConfig>) => Promise<void>;
  setPanel: (open: boolean, tab?: PanelTab) => void;
}

let socket: SessionSocket | undefined;
let filterRunId: string | null = null;

export const useStore = create<Store>((set, get) => {
  const patchChat = (sid: string, fn: (c: ChatState) => ChatState) =>
    set((s) => ({ chats: { ...s.chats, [sid]: fn(s.chats[sid] ?? emptyChat()) } }));

  // Streamed tokens arrive in bursts. Applying one store update per event re-renders the whole timeline per
  // token (and trips React's nested-update guard), so events are queued and reduced once per ~30 ms tick.
  let queue: ServerEvent[] = [];
  let timer: ReturnType<typeof setTimeout> | null = null;
  const flush = (sid: string) => {
    timer = null;
    const batch = queue;
    queue = [];
    if (!batch.length) return;
    patchChat(sid, (c) => batch.reduce(applyEvent, c));
    if (batch.some((e) => e.type === "session_updated" || e.type === "run_finished")) get().refreshSessions();
  };

  const onEvent = (sid: string) => (e: ServerEvent) => {
    if (filterRunId && e.run_id && e.run_id !== filterRunId) return; // replay of older runs
    if (e.type === "run_finished" && e.run_id === filterRunId) filterRunId = null;
    queue.push(e);
    if (e.type === "run_finished" || e.type === "approval_request") { if (timer) clearTimeout(timer); flush(sid); return; }
    timer ??= setTimeout(() => flush(sid), 30);
  };

  return {
    connected: false, sessions: [], chats: {}, panelOpen: true, panelTab: "files", booting: true,

    async boot() {
      try {
        const [config, health, sessions] = await Promise.all([api.config(), api.health(), api.sessions()]);
        set({ config, health, sessions, booting: false, fatal: undefined });
        if (sessions.length) await get().select(sessions[0].id);
      } catch (err: any) {
        set({ booting: false, fatal: err?.status === 401 ? "auth" : String(err?.message ?? err) });
      }
      setInterval(() => api.health().then((health) => set({ health })).catch(() => undefined), 15000);
    },

    async refreshSessions() {
      set({ sessions: await api.sessions() });
    },

    async select(id) {
      socket?.close();
      filterRunId = null;
      if (timer) { clearTimeout(timer); timer = null; }
      queue = [];
      set({ activeId: id, connected: false });
      const [session, msgs, st] = await Promise.all([api.session(id), api.messages(id), api.state(id)]);
      const items = itemsFromMessages(msgs, st.running ? st.run_id : null);
      const chat: ChatState = {
        ...emptyChat(), items, todos: st.todos, tasks: st.tasks, running: st.running, runId: st.run_id, usage: session.usage,
        lastSeq: st.last_seq, context: { tokens: 0, budget: get().config?.model.context_budget ?? 0 },
      };
      set((s) => ({ chats: { ...s.chats, [id]: chat }, sessions: s.sessions.map((x) => (x.id === id ? { ...x, ...session } : x)) }));
      let since = st.last_seq;
      if (st.running) { filterRunId = st.run_id; since = 0; }
      socket = new SessionSocket(id, since, onEvent(id), (connected) => set({ connected }));
    },

    async newSession(config) {
      const s = await api.createSession(config);
      set((st) => ({ sessions: [s, ...st.sessions] }));
      await get().select(s.id);
    },

    async removeSession(id) {
      await api.deleteSession(id);
      const rest = get().sessions.filter((s) => s.id !== id);
      set({ sessions: rest });
      if (get().activeId === id) {
        socket?.close();
        if (rest.length) await get().select(rest[0].id);
        else set({ activeId: undefined });
      }
    },

    async rename(id, title) {
      await api.patchSession(id, { title });
      set((s) => ({ sessions: s.sessions.map((x) => (x.id === id ? { ...x, title } : x)) }));
    },

    async send(text) {
      if (!text.trim()) return;
      if (!get().activeId) await get().newSession(); // first message from the empty welcome screen
      const sid = get().activeId;
      if (!sid) return;
      const cfg = get().sessions.find((s) => s.id === sid)?.config;
      patchChat(sid, (c) => ({ ...c, running: true, error: undefined, status: "Sending…" }));
      try {
        // WebSocket path keeps everything on one channel; REST is the fallback.
        const ok = socket?.send({ type: "send", content: text, options: cfg ? { mode: cfg.mode, thinking: cfg.thinking, approval: cfg.approval, mcp: cfg.mcp, memory: cfg.memory } : {} });
        if (!ok) await api.send(sid, text, cfg);
      } catch (err: any) {
        patchChat(sid, (c) => ({ ...c, running: false, error: String(err?.message ?? err) }));
      }
    },

    async cancel() {
      const sid = get().activeId;
      if (!sid) return;
      if (!socket?.send({ type: "cancel" })) await api.cancel(sid);
    },

    async approve(approvalId, approve) {
      const sid = get().activeId;
      if (!sid) return;
      if (!socket?.send({ type: "approval", approval_id: approvalId, approve })) await api.approve(sid, approvalId, approve);
    },

    async patchConfig(patch) {
      const sid = get().activeId;
      if (!sid) return;
      set((s) => ({ sessions: s.sessions.map((x) => (x.id === sid ? { ...x, config: { ...x.config, ...patch } } : x)) }));
      await api.patchSession(sid, { config: patch });
    },

    setPanel(open, tab) {
      set((s) => ({ panelOpen: open, panelTab: tab ?? s.panelTab }));
    },
  };
});

export const useChat = (): ChatState => {
  const id = useStore((s) => s.activeId);
  const chat = useStore((s) => (id ? s.chats[id] : undefined));
  return chat ?? emptyChat();
};

export const useActiveSession = (): Session | undefined =>
  useStore((s) => s.sessions.find((x) => x.id === s.activeId));
