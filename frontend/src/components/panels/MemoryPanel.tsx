import { useCallback, useEffect, useState } from "react";
import { Button, Input, message, Popconfirm, Select, Tag } from "antd";
import { DeleteOutlined, PlusOutlined } from "@ant-design/icons";
import { api } from "../../api/client";
import type { MemoryItem } from "../../api/types";
import { useActiveSession, useChat, useStore } from "../../store/useStore";

const BLOCKS: { key: string; label: string; hint: string }[] = [
  { key: "user", label: "About the user", hint: "Name, role, preferences…" },
  { key: "project", label: "Project", hint: "Goals, stack, conventions…" },
  { key: "notes", label: "Scratchpad", hint: "Working notes the agent keeps" },
];

export function MemoryPanel() {
  const sid = useStore((s) => s.activeId)!;
  const session = useActiveSession();
  const chat = useChat();
  const refresh = useStore((s) => s.refreshSessions);
  const [items, setItems] = useState<MemoryItem[]>([]);
  const [text, setText] = useState("");
  const [kind, setKind] = useState("fact");
  const [scope, setScope] = useState<"global" | "session">("global");
  const [core, setCore] = useState<Record<string, string>>({});
  const load = useCallback(() => { api.memories(sid).then(setItems).catch(() => undefined); api.session(sid).then((s) => setCore(s.core_memory)).catch(() => undefined); }, [sid]);
  useEffect(load, [load, chat.memoryTick]);
  void session; void refresh;
  return (
    <div className="panel-pad">
      <h4>Core memory <span className="muted small">always in the agent's context</span></h4>
      {BLOCKS.map((b) => (
        <div key={b.key} className="core-block">
          <label>{b.label}</label>
          <Input.TextArea
            autoSize={{ minRows: 2, maxRows: 6 }} maxLength={1200} placeholder={b.hint} value={core[b.key] ?? ""}
            onChange={(e) => setCore({ ...core, [b.key]: e.target.value })}
            onBlur={() => api.setCore(sid, b.key, core[b.key] ?? "").catch((e) => message.error(e.message))}
          />
        </div>
      ))}
      <h4 style={{ marginTop: 18 }}>Long-term memory <span className="muted small">{items.length} items</span></h4>
      <div className="mem-add">
        <Input.TextArea autoSize={{ minRows: 1, maxRows: 4 }} placeholder="Teach the agent something to remember…" value={text} onChange={(e) => setText(e.target.value)} />
        <div className="mem-add-row">
          <Select size="small" value={kind} onChange={setKind} options={["fact", "preference", "project", "lesson"].map((v) => ({ value: v, label: v }))} />
          <Select size="small" value={scope} onChange={setScope} options={[{ value: "global", label: "all sessions" }, { value: "session", label: "this session" }]} />
          <Button size="small" type="primary" icon={<PlusOutlined />} disabled={!text.trim()} onClick={async () => { await api.addMemory(text, kind, scope, sid); setText(""); load(); }}>Add</Button>
        </div>
      </div>
      {items.map((m) => (
        <div key={m.id} className="mem">
          <div className="mem-text">{m.content}</div>
          <div className="mem-meta">
            <Tag bordered={false}>{m.kind}</Tag>
            <Tag bordered={false} color={m.session_id ? "default" : "purple"}>{m.session_id ? "session" : "global"}</Tag>
            <span className="muted small">{m.source}</span>
            <Popconfirm title="Forget this memory?" onConfirm={async () => { await api.deleteMemory(m.id); load(); }}><Button type="text" size="small" danger icon={<DeleteOutlined />} aria-label="Forget" /></Popconfirm>
          </div>
        </div>
      ))}
      {!items.length && <div className="muted pad">Nothing stored yet. The agent saves lessons and preferences as it works.</div>}
    </div>
  );
}
