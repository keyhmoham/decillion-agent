import { useState } from "react";
import { Tag, Tooltip } from "antd";
import { BulbOutlined, DownOutlined, RightOutlined, WarningOutlined } from "@ant-design/icons";
import type { TimelineItem } from "../../api/types";
import { Markdown } from "../Markdown";

type A = Extract<TimelineItem, { kind: "assistant" }>;

export function AssistantBlock({ a, roleColor }: { a: A; roleColor?: string }) {
  const [open, setOpen] = useState<boolean | null>(null);
  const expanded = open ?? (a.streaming && !a.content);
  const hasText = a.content.trim().length > 0;
  if (!a.reasoning && !hasText && !a.streaming) return null;
  return (
    <div className="assistant" data-testid="assistant">
      {a.agent !== "main" && (
        <div className="agent-line">
          <span className="agent-dot" style={{ background: roleColor ?? "#6366f1" }} />
          <span>{a.agent}</span>
        </div>
      )}
      {(a.reasoning || (a.streaming && !hasText)) && (
        <div className={`thinking ${a.streaming && !hasText ? "live" : ""}`}>
          <button className="think-head" onClick={() => setOpen(!expanded)}>
            {expanded ? <DownOutlined /> : <RightOutlined />}
            <BulbOutlined />
            <span>{a.streaming && !hasText ? "Thinking…" : "Thought process"}</span>
            {a.flags?.thinking_forced_closed && (
              <Tooltip title={a.flags.thinking_degenerate ? "The model started repeating itself; reasoning was cut and it was told to answer." : "Reasoning budget reached; the model was told to answer."}>
                <Tag color="warning" bordered={false} icon={<WarningOutlined />}>cut short</Tag>
              </Tooltip>
            )}
            {a.flags?.nothink_fallback && <Tag bordered={false}>retried without thinking</Tag>}
          </button>
          {expanded && <div className="think-body">{a.reasoning || "…"}</div>}
        </div>
      )}
      {hasText && <Markdown text={a.content} />}
      {a.streaming && hasText && <span className="cursor" />}
    </div>
  );
}
