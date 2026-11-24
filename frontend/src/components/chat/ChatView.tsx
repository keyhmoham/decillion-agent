import { useState } from "react";
import { Button, Collapse, Input, Progress, Tag, Tooltip } from "antd";
import { CodeOutlined, EditOutlined, MenuUnfoldOutlined, MenuFoldOutlined, ThunderboltFilled, TeamOutlined, ApiOutlined, LoadingOutlined } from "@ant-design/icons";
import { useActiveSession, useChat, useStore } from "../../store/useStore";
import { fmtTokens } from "../../lib/format";
import { Composer } from "./Composer";
import { TodoList } from "./Blocks";
import { Timeline } from "./Timeline";

const EXAMPLES = [
  { icon: <CodeOutlined />, title: "Build a CLI app", text: "Create a Python CLI todo app with add/list/done commands stored in JSON, write pytest tests for it, run them and show me the results." },
  { icon: <ThunderboltFilled />, title: "Analyse & visualise", text: "Generate a synthetic CSV of 200 monthly sales records for 4 regions, compute totals and growth per region, and show me a chart and a summary table." },
  { icon: <TeamOutlined />, title: "Team project", text: "Use the agent team: build a small Flask API for notes with CRUD endpoints, write tests for it, and write a README. Verify everything runs." },
  { icon: <ApiOutlined />, title: "Research", text: "Search the web for the three most popular Python web frameworks today and show a comparison table with their main trade-offs." },
];

function Header() {
  const s = useActiveSession()!;
  const chat = useChat();
  const rename = useStore((st) => st.rename);
  const config = useStore((st) => st.config);
  const health = useStore((st) => st.health);
  const connected = useStore((st) => st.connected);
  const panelOpen = useStore((st) => st.panelOpen);
  const setPanel = useStore((st) => st.setPanel);
  const [editing, setEditing] = useState(false);
  const budget = chat.context.budget || config?.model.context_budget || 1;
  const pct = Math.min(100, Math.round((chat.context.tokens / budget) * 100));
  const modelOk = health?.model?.ok;
  return (
    <header className="chat-header">
      <div className="title-wrap">
        {editing ? (
          <Input size="small" autoFocus defaultValue={s.title} onBlur={(e) => { rename(s.id, e.target.value || s.title); setEditing(false); }} onPressEnter={(e) => { rename(s.id, (e.target as HTMLInputElement).value || s.title); setEditing(false); }} />
        ) : (
          <>
            <h1 title={s.title}>{s.title}</h1>
            <Button type="text" size="small" icon={<EditOutlined />} onClick={() => setEditing(true)} aria-label="Rename" />
          </>
        )}
      </div>
      <div className="header-chips">
        {chat.mode && <Tooltip title={chat.modeReason}><Tag color={chat.mode === "team" ? "purple" : "blue"} bordered={false}>{chat.mode}</Tag></Tooltip>}
        <Tooltip title={`Model: ${config?.model.tag}\n${modelOk ? "reachable" : "unreachable - check the model server"}`}>
          <Tag bordered={false} color={modelOk ? "success" : "error"}>{config?.model.parameters} · {config?.model.quantization}</Tag>
        </Tooltip>
        {!connected && <Tag color="warning" bordered={false}>reconnecting…</Tag>}
        <Tooltip title={`Prompt: ${chat.context.tokens} / ${budget} tokens before the history is compacted`}>
          <div className="ctx-meter"><Progress type="circle" percent={pct} size={22} strokeWidth={14} showInfo={false} strokeColor={pct > 85 ? "#f87171" : pct > 60 ? "#fbbf24" : "#818cf8"} /><span>{fmtTokens(chat.context.tokens)}</span></div>
        </Tooltip>
        <Button type="text" icon={panelOpen ? <MenuFoldOutlined /> : <MenuUnfoldOutlined />} onClick={() => setPanel(!panelOpen)} aria-label="Toggle side panel" />
      </div>
    </header>
  );
}

function Welcome() {
  const send = useStore((s) => s.send);
  const config = useStore((s) => s.config);
  return (
    <div className="welcome">
      <div className="logo-big">D</div>
      <h2>What should we build?</h2>
      <p className="muted">
        Every chat gets its own Docker computer. Powered by <b>{config?.model.tag}</b> with planning, tools, MCP, memory and agent teams.
      </p>
      <div className="examples">
        {EXAMPLES.map((e) => (
          <button key={e.title} className="example" onClick={() => send(e.text)}>
            <span className="example-icon">{e.icon}</span>
            <b>{e.title}</b>
            <span>{e.text}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

export function ChatView() {
  const s = useActiveSession();
  const chat = useChat();
  if (!s) return <div className="welcome"><Welcome /></div>;
  const doneTodos = chat.todos.filter((t) => t.status === "completed").length;
  return (
    <section className="chat">
      <Header />
      {chat.todos.length > 0 && (
        <Collapse
          size="small" className="todo-strip" ghost
          items={[{
            key: "todos",
            label: <span className="todo-strip-label">{chat.running && <LoadingOutlined spin />} Plan · {doneTodos}/{chat.todos.length} steps done</span>,
            children: <TodoList todos={chat.todos} compact />,
          }]}
        />
      )}
      {chat.items.length === 0 && !chat.running ? <Welcome /> : <Timeline items={chat.items} running={chat.running} />}
      {chat.running && chat.status && <div className="status-line"><LoadingOutlined spin /> {chat.status}</div>}
      <Composer />
    </section>
  );
}
