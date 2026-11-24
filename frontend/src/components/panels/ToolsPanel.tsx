import { useEffect, useState } from "react";
import { Alert, Switch, Tag } from "antd";
import { api } from "../../api/client";
import { useActiveSession, useChat, useStore } from "../../store/useStore";

export function ToolsPanel() {
  const session = useActiveSession()!;
  const config = useStore((s) => s.config);
  const patch = useStore((s) => s.patchConfig);
  const chat = useChat();
  const [servers, setServers] = useState<{ name: string; tools: string[]; error: string | null }[]>([]);
  useEffect(() => { api.state(session.id).then((s) => setServers(s.mcp)); }, [session.id, chat.running]);
  const enabled = new Set(session.config.mcp);
  const tokens = (config?.mcp_catalog ?? []).filter((m) => enabled.has(m.id)).reduce((a, m) => a + m.approx_prompt_tokens, 0);
  const toggle = (id: string, on: boolean) => patch({ mcp: on ? [...enabled, id] : [...enabled].filter((x) => x !== id) });
  return (
    <div className="panel-pad">
      <h4>MCP servers <span className="muted small">run inside the session sandbox</span></h4>
      <Alert type={tokens > 1500 ? "warning" : "info"} showIcon banner className="mb"
        message={`≈${tokens} extra prompt tokens. A 4B model is most accurate with a short tool list - enable only what you need. Changes apply on the next message.`} />
      {(config?.mcp_catalog ?? []).map((m) => {
        const live = servers.find((s) => s.name === m.id);
        return (
          <div key={m.id} className="mcp-row">
            <div className="mcp-main">
              <div><b>{m.name}</b> <Tag bordered={false}>{m.category}</Tag>{m.requires.length > 0 && !m.available && <Tag color="warning" bordered={false}>needs {m.requires.join(", ")}</Tag>}</div>
              <div className="muted small">{m.description}</div>
              {live && !live.error && <div className="small good-text">connected · {live.tools.join(", ")}</div>}
              {live?.error && <div className="small bad-text">{live.error}</div>}
            </div>
            <Switch size="small" checked={enabled.has(m.id)} disabled={!m.available} onChange={(v) => toggle(m.id, v)} />
          </div>
        );
      })}
    </div>
  );
}
