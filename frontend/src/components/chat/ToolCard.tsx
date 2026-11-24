import { useState } from "react";
import { Spin, Tag } from "antd";
import {
  CheckCircleFilled, CloseCircleFilled, CodeOutlined, EditOutlined, FileAddOutlined, FileSearchOutlined, FolderOpenOutlined,
  GlobalOutlined, SearchOutlined, DownOutlined, RightOutlined, ToolOutlined, DatabaseOutlined, UnorderedListOutlined, LoadingOutlined,
  ClusterOutlined, BulbOutlined, BarChartOutlined,
} from "@ant-design/icons";
import hljs from "highlight.js/lib/common";
import type { TimelineItem } from "../../api/types";
import { extToLang, stripAnsi } from "../../lib/format";

type Tool = Extract<TimelineItem, { kind: "tool" }>;

const ICON: Record<string, React.ReactNode> = {
  bash: <CodeOutlined />, bash_job: <CodeOutlined />, write_file: <FileAddOutlined />, edit_file: <EditOutlined />, read_file: <FileSearchOutlined />,
  list_dir: <FolderOpenOutlined />, glob: <SearchOutlined />, grep: <SearchOutlined />, web_search: <GlobalOutlined />, web_fetch: <GlobalOutlined />,
  write_todos: <UnorderedListOutlined />, update_todo: <UnorderedListOutlined />, memory_search: <DatabaseOutlined />, memory_save: <DatabaseOutlined />,
  memory_forget: <DatabaseOutlined />, core_memory_update: <DatabaseOutlined />, delegate_task: <ClusterOutlined />, handoff: <ClusterOutlined />,
  show_table: <BarChartOutlined />, show_chart: <BarChartOutlined />, show_stats: <BarChartOutlined />, show_diagram: <BarChartOutlined />, show_html: <BarChartOutlined />,
};

function summary(t: Tool): string {
  const a = t.args as Record<string, any>;
  switch (t.name) {
    case "bash": return String(a.command ?? "").split("\n")[0];
    case "write_file": case "edit_file": case "read_file": return String(a.path ?? "");
    case "list_dir": case "glob": return String(a.pattern ?? a.path ?? ".");
    case "grep": return `${a.pattern ?? ""} ${a.path ?? ""}`;
    case "web_search": return String(a.query ?? "");
    case "web_fetch": return String(a.url ?? "");
    case "write_todos": return `${(a.todos as unknown[] | undefined)?.length ?? 0} steps`;
    case "update_todo": return `step ${a.index} → ${a.status}`;
    case "delegate_task": return `${a.role}: ${String(a.task ?? "").slice(0, 80)}`;
    case "memory_search": return String(a.query ?? "");
    case "memory_save": return String(a.content ?? "").slice(0, 80);
    default: return Object.values(a).map((v) => (typeof v === "string" ? v : JSON.stringify(v))).join(" ").slice(0, 100);
  }
}

function Code({ text, lang }: { text: string; lang?: string }) {
  let html: string;
  try {
    html = lang && hljs.getLanguage(lang) ? hljs.highlight(text, { language: lang }).value : hljs.highlightAuto(text.slice(0, 20000)).value;
  } catch { html = text.replace(/[&<]/g, (c) => (c === "&" ? "&amp;" : "&lt;")); }
  return <pre className="toolpre hljs"><code dangerouslySetInnerHTML={{ __html: html }} /></pre>;
}

function Diff({ text }: { text: string }) {
  const lines = text.split("\n");
  return (
    <pre className="toolpre diff">
      {lines.map((l, i) => (
        <div key={i} className={l.startsWith("+") && !l.startsWith("+++") ? "add" : l.startsWith("-") && !l.startsWith("---") ? "del" : l.startsWith("@@") ? "hunk" : ""}>{l || " "}</div>
      ))}
    </pre>
  );
}

function Body({ t }: { t: Tool }) {
  const a = t.args as Record<string, any>;
  const out = stripAnsi(t.status === "running" ? t.output : t.result ?? t.output);
  switch (t.name) {
    case "bash":
      return (
        <>
          <pre className="toolpre term"><span className="prompt">$ </span>{String(a.command)}</pre>
          {out && <pre className={`toolpre term out ${t.ok === false ? "err" : ""}`}>{out}</pre>}
        </>
      );
    case "write_file":
      return <><div className="toolmeta">{String(a.path)}</div><Code text={String(a.content ?? "")} lang={extToLang(String(a.path))} /><div className="toolres">{t.result}</div></>;
    case "edit_file": {
      const diff = (t.result ?? "").split("\n").slice(1).join("\n");
      return <><div className="toolmeta">{String(a.path)}</div>{diff ? <Diff text={diff} /> : <pre className="toolpre">{t.result}</pre>}</>;
    }
    case "read_file":
      return <pre className="toolpre">{out}</pre>;
    case "write_todos": case "update_todo":
      return <pre className="toolpre">{out}</pre>;
    default:
      return (
        <>
          <pre className="toolpre args">{JSON.stringify(a, null, 2)}</pre>
          {out && <pre className={`toolpre ${t.ok === false ? "err" : ""}`}>{out}</pre>}
        </>
      );
  }
}

export function ToolCard({ t }: { t: Tool }) {
  const autoOpen = t.name === "bash" || t.status === "error";
  const [open, setOpen] = useState<boolean | null>(null);
  const expanded = open ?? (autoOpen && (t.status === "running" || t.name === "bash"));
  const icon = ICON[t.name] ?? <ToolOutlined />;
  return (
    <div className={`toolcard st-${t.status}`} data-testid="toolcard">
      <button className="toolhead" onClick={() => setOpen(!expanded)} aria-expanded={expanded}>
        <span className="tool-caret">{expanded ? <DownOutlined /> : <RightOutlined />}</span>
        <span className="tool-icon">{icon}</span>
        <span className="tool-name">{t.name}</span>
        <span className="tool-sum">{summary(t)}</span>
        {t.agent !== "main" && <Tag bordered={false} className="agent-tag">{t.agent}</Tag>}
        <span className="tool-status">
          {t.status === "running" ? <Spin indicator={<LoadingOutlined spin />} size="small" /> :
            t.status === "error" ? <CloseCircleFilled className="bad" /> : <CheckCircleFilled className="good" />}
          {t.duration !== undefined && <span className="dur">{t.duration < 1 ? `${Math.round(t.duration * 1000)}ms` : `${t.duration.toFixed(1)}s`}</span>}
        </span>
      </button>
      {expanded && <div className="toolbody"><Body t={t} /></div>}
      {t.name === "bash" && !expanded && t.result && <div className="tool-peek">{stripAnsi(t.result).split("\n").filter(Boolean).slice(-1)[0]?.slice(0, 140)}</div>}
    </div>
  );
}

export { BulbOutlined };
