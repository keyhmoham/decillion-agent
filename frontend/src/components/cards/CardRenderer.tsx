import { lazy, Suspense, useEffect, useRef, useState } from "react";
import { Table, Tag, Typography } from "antd";
import { ArrowDownOutlined, ArrowUpOutlined, BarChartOutlined, TableOutlined, ApartmentOutlined, CodeOutlined, PictureOutlined, InfoCircleOutlined, DashboardOutlined } from "@ant-design/icons";
import type { Card } from "../../api/types";
import { fileUrl } from "../../api/client";
import { useStore } from "../../store/useStore";
import { Markdown } from "../Markdown";

const ChartCard = lazy(() => import("./ChartCard"));

const ICONS: Record<Card["kind"], React.ReactNode> = {
  table: <TableOutlined />, chart: <BarChartOutlined />, stats: <DashboardOutlined />, diagram: <ApartmentOutlined />,
  html: <CodeOutlined />, image: <PictureOutlined />, callout: <InfoCircleOutlined />,
};

function TableCard({ data }: { data: { columns: string[]; rows: any[][]; caption?: string } }) {
  const columns = data.columns.map((c, i) => ({
    title: c, dataIndex: i, key: i, ellipsis: true,
    render: (v: any) => (v === null || v === undefined ? <span className="muted">—</span> : typeof v === "boolean" ? <Tag color={v ? "green" : "red"}>{String(v)}</Tag> : String(v)),
    sorter: (a: any, b: any) => (typeof a[i] === "number" && typeof b[i] === "number" ? a[i] - b[i] : String(a[i]).localeCompare(String(b[i]))),
  }));
  const rows = data.rows.map((r, k) => ({ key: k, ...r }));
  return (
    <>
      <Table size="small" columns={columns} dataSource={rows} pagination={rows.length > 10 ? { pageSize: 10, size: "small", showSizeChanger: false } : false} scroll={{ x: "max-content" }} />
      {data.caption && <div className="card-caption">{data.caption}</div>}
    </>
  );
}

function StatsCard({ data }: { data: { items: { label: string; value: string; delta?: string | null; status?: string }[] } }) {
  return (
    <div className="stats-grid">
      {data.items.map((it, i) => {
        const up = it.delta?.trim().startsWith("+");
        const down = it.delta?.trim().startsWith("-");
        return (
          <div key={i} className={`stat stat-${it.status ?? "neutral"}`}>
            <div className="stat-label">{it.label}</div>
            <div className="stat-value">{it.value}</div>
            {it.delta && <div className="stat-delta">{up ? <ArrowUpOutlined /> : down ? <ArrowDownOutlined /> : null} {it.delta}</div>}
          </div>
        );
      })}
    </div>
  );
}

let mermaidReady: Promise<typeof import("mermaid").default> | null = null;
function loadMermaid() {
  mermaidReady ??= import("mermaid").then((m) => {
    m.default.initialize({ startOnLoad: false, theme: "dark", securityLevel: "strict", fontFamily: "inherit" });
    return m.default;
  });
  return mermaidReady;
}

function DiagramCard({ data }: { data: { mermaid: string } }) {
  const [svg, setSvg] = useState("");
  const [err, setErr] = useState("");
  useEffect(() => {
    let live = true;
    loadMermaid()
      .then((m) => m.render(`mmd-${Math.random().toString(36).slice(2)}`, data.mermaid))
      .then((r) => live && setSvg(r.svg))
      .catch((e) => live && setErr(String(e?.message ?? e)));
    return () => { live = false; };
  }, [data.mermaid]);
  if (err) return <><Typography.Text type="warning">Could not render diagram: {err.split("\n")[0]}</Typography.Text><pre className="plain-pre">{data.mermaid}</pre></>;
  if (!svg) return <div className="muted pad">Rendering diagram…</div>;
  return <div className="diagram" dangerouslySetInnerHTML={{ __html: svg }} />;
}

function HtmlCard({ data }: { data: { html: string; height: number } }) {
  // Opaque origin + no network: a fully isolated canvas for model-written visuals.
  const doc = `<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data:;"><style>body{margin:0;padding:12px;font-family:system-ui,sans-serif;color:#e5e7eb;background:#0b0f17}</style></head><body>${data.html}</body></html>`;
  return <iframe className="html-card" sandbox="allow-scripts" srcDoc={doc} style={{ height: data.height }} title="visual" />;
}

function ImageCard({ data }: { data: { path: string; caption?: string } }) {
  const sid = useStore((s) => s.activeId)!;
  return <img className="img-card" src={fileUrl(sid, data.path)} alt={data.caption ?? data.path} />;
}

function CalloutCard({ data }: { data: { body: string; level: string; items: string[] } }) {
  return (
    <div className={`callout callout-${data.level}`}>
      <Markdown text={data.body} />
      {data.items?.length > 0 && <ul>{data.items.map((i, k) => <li key={k}>{i}</li>)}</ul>}
    </div>
  );
}

export function CardRenderer({ card, agent }: { card: Card; agent?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  let body: React.ReactNode;
  switch (card.kind) {
    case "table": body = <TableCard data={card.data} />; break;
    case "chart": body = <Suspense fallback={<div className="muted pad">Loading chart…</div>}><ChartCard data={card.data} /></Suspense>; break;
    case "stats": body = <StatsCard data={card.data} />; break;
    case "diagram": body = <DiagramCard data={card.data} />; break;
    case "html": body = <HtmlCard data={card.data} />; break;
    case "image": body = <ImageCard data={card.data} />; break;
    case "callout": body = <CalloutCard data={card.data} />; break;
    default: body = <pre className="plain-pre">{JSON.stringify(card.data, null, 2)}</pre>;
  }
  return (
    <div className={`vcard vcard-${card.kind}`} ref={ref} data-testid={`card-${card.kind}`}>
      <div className="vcard-head">
        <span className="vcard-icon">{ICONS[card.kind]}</span>
        <span className="vcard-title">{card.title}</span>
        {agent && agent !== "main" && <Tag bordered={false}>{agent}</Tag>}
      </div>
      <div className="vcard-body">{body}</div>
    </div>
  );
}
