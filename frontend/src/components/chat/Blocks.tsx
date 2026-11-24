import { useState } from "react";
import { Button, Progress, Tag } from "antd";
import {
  CheckCircleFilled, ClockCircleOutlined, CloseCircleFilled, LoadingOutlined, MinusCircleOutlined, SafetyCertificateOutlined, WarningFilled, InfoCircleFilled,
} from "@ant-design/icons";
import type { TeamTask, TimelineItem, Todo } from "../../api/types";
import { useStore } from "../../store/useStore";

export function UserBubble({ content }: { content: string }) {
  return <div className="user-msg" data-testid="user-msg"><div className="user-bubble">{content}</div></div>;
}

export function Notice({ level, text }: { level: string; text: string }) {
  const icon = level === "error" ? <CloseCircleFilled className="bad" /> : level === "warn" ? <WarningFilled className="warn" /> : <InfoCircleFilled className="info" />;
  return <div className={`notice notice-${level}`}>{icon}<span>{text}</span></div>;
}

export function ApprovalCard({ it }: { it: Extract<TimelineItem, { kind: "approval" }> }) {
  const approve = useStore((s) => s.approve);
  const a = it.args as Record<string, any>;
  const preview = it.tool === "bash" ? String(a.command) : it.tool === "write_file" ? `${a.path}\n\n${String(a.content ?? "").slice(0, 600)}` : JSON.stringify(a, null, 2);
  return (
    <div className={`approval ap-${it.status}`}>
      <div className="approval-head"><SafetyCertificateOutlined /> <b>Permission requested</b> <Tag bordered={false}>{it.tool}</Tag></div>
      <pre className="toolpre">{preview}</pre>
      {it.status === "pending" ? (
        <div className="approval-actions">
          <Button type="primary" size="small" onClick={() => approve(it.approvalId, true)}>Allow</Button>
          <Button size="small" danger onClick={() => approve(it.approvalId, false)}>Deny</Button>
        </div>
      ) : <div className="approval-result">{it.status === "approved" ? "Allowed" : "Denied"}</div>}
    </div>
  );
}

export function TodoList({ todos, compact }: { todos: Todo[]; compact?: boolean }) {
  const done = todos.filter((t) => t.status === "completed").length;
  return (
    <div className="todos">
      {!compact && <Progress percent={todos.length ? Math.round((done / todos.length) * 100) : 0} size="small" status="active" />}
      {todos.map((t, i) => (
        <div key={i} className={`todo todo-${t.status}`}>
          {t.status === "completed" ? <CheckCircleFilled className="good" /> : t.status === "in_progress" ? <LoadingOutlined className="info" spin /> : <ClockCircleOutlined className="muted" />}
          <span>{t.content}</span>
        </div>
      ))}
      {!todos.length && <div className="muted pad">No plan yet. The agent writes one for multi-step tasks.</div>}
    </div>
  );
}

const STATUS_ICON: Record<TeamTask["status"], React.ReactNode> = {
  pending: <ClockCircleOutlined className="muted" />, running: <LoadingOutlined className="info" spin />, done: <CheckCircleFilled className="good" />,
  failed: <CloseCircleFilled className="bad" />, blocked: <MinusCircleOutlined className="warn" />,
};

export function TaskBoard({ tasks, summary }: { tasks: TeamTask[]; summary?: string }) {
  const roles = useStore((s) => s.config?.roles ?? []);
  const [open, setOpen] = useState<string | null>(null);
  const color = (r: string) => roles.find((x) => x.name === r)?.color ?? "#6366f1";
  return (
    <div className="taskboard" data-testid="taskboard">
      {summary && <div className="plan-summary">{summary}</div>}
      {tasks.map((t) => (
        <div key={t.key} className={`task task-${t.status}`}>
          <button className="task-row" onClick={() => setOpen(open === t.key ? null : t.key)}>
            {STATUS_ICON[t.status]}
            <span className="task-key">{t.key}</span>
            <span className="task-title">{t.title}</span>
            <Tag bordered={false} style={{ color: color(t.role), background: `${color(t.role)}22` }}>{t.role}</Tag>
            {t.attempts > 1 && <Tag bordered={false}>try {t.attempts}</Tag>}
          </button>
          {t.depends_on.length > 0 && <div className="task-deps">after {t.depends_on.join(", ")}{t.handoff_from ? ` · handed off from ${t.handoff_from}` : ""}</div>}
          {open === t.key && (
            <div className="task-detail">
              <div className="muted">{t.description}</div>
              {t.result && <pre className="toolpre">{t.result}</pre>}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

export function PlanCard({ it }: { it: Extract<TimelineItem, { kind: "plan" }> }) {
  const done = it.tasks.filter((t) => t.status === "done").length;
  return (
    <div className="vcard" data-testid="plan-card">
      <div className="vcard-head"><span className="vcard-title">Team plan</span><Tag bordered={false}>{done}/{it.tasks.length} done</Tag></div>
      <div className="vcard-body"><TaskBoard tasks={it.tasks} summary={it.summary} /></div>
    </div>
  );
}
