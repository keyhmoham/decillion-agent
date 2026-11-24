import { useRef, useState } from "react";
import { Button, Dropdown, message, Segmented, Select, Switch, Tooltip } from "antd";
import { ArrowUpOutlined, PaperClipOutlined, StopFilled, ThunderboltOutlined, TeamOutlined, RobotOutlined } from "@ant-design/icons";
import { api } from "../../api/client";
import type { Mode } from "../../api/types";
import { useActiveSession, useChat, useStore } from "../../store/useStore";

export function Composer() {
  const session = useActiveSession();
  const chat = useChat();
  const send = useStore((s) => s.send);
  const cancel = useStore((s) => s.cancel);
  const patch = useStore((s) => s.patchConfig);
  const [text, setText] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);
  if (!session) return null;
  const cfg = session.config;

  const submit = () => {
    const t = text.trim();
    if (!t || chat.running) return;
    setText("");
    send(t);
  };

  const onFiles = async (files: FileList | null) => {
    if (!files?.length) return;
    try {
      const res = await api.upload(session.id, Array.from(files));
      message.success(`Uploaded ${res.saved.length} file(s) to the workspace`);
      setText((t) => `${t}${t ? "\n" : ""}I uploaded: ${res.saved.map((p) => p.split("/").pop()).join(", ")}. `);
    } catch (e: any) { message.error(e.message); }
  };

  return (
    <div className="composer-wrap">
      <div className="composer">
        <textarea
          className="composer-input"
          placeholder={chat.running ? "The agent is working… you can stop it at any time" : "Ask the agent to build, fix, research or analyse something…"}
          value={text}
          rows={Math.min(8, Math.max(2, text.split("\n").length))}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); submit(); } }}
          aria-label="Message"
        />
        <div className="composer-bar">
          <div className="composer-opts">
            <Segmented
              size="small"
              value={cfg.mode}
              onChange={(v) => patch({ mode: v as Mode })}
              options={[
                { value: "auto", label: <span title="The agent decides between solo and team"><ThunderboltOutlined /> Auto</span> },
                { value: "agent", label: <span title="One agent with planning, tools and memory"><RobotOutlined /> Agent</span> },
                { value: "team", label: <span title="Orchestrator + specialist agents working in parallel"><TeamOutlined /> Team</span> },
              ]}
            />
            <Tooltip title="Let the model reason before acting (slower, more accurate)">
              <span className="opt"><Switch size="small" checked={cfg.thinking} onChange={(v) => patch({ thinking: v })} /> Thinking</span>
            </Tooltip>
            <Select
              size="small" variant="borderless" value={cfg.approval} onChange={(v) => patch({ approval: v })} style={{ width: 140 }} popupMatchSelectWidth={false}
              options={[{ value: "auto", label: "Auto-run tools" }, { value: "ask", label: "Ask before changes" }]}
            />
            <input ref={fileRef} type="file" multiple hidden onChange={(e) => { onFiles(e.target.files); e.target.value = ""; }} />
            <Dropdown menu={{ items: [{ key: "up", label: "Upload files to workspace", onClick: () => fileRef.current?.click() }] }} trigger={["click"]}>
              <Button type="text" size="small" icon={<PaperClipOutlined />} aria-label="Attach" />
            </Dropdown>
          </div>
          {chat.running ? (
            <Button type="primary" danger shape="circle" icon={<StopFilled />} onClick={() => cancel()} aria-label="Stop" />
          ) : (
            <Button type="primary" shape="circle" icon={<ArrowUpOutlined />} disabled={!text.trim()} onClick={submit} aria-label="Send" />
          )}
        </div>
      </div>
      <div className="composer-hint">Enter to send · Shift+Enter for a new line · runs inside an isolated Docker sandbox</div>
    </div>
  );
}
