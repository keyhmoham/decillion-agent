import { useEffect, useMemo, useRef } from "react";
import type { TimelineItem } from "../../api/types";
import { useStore } from "../../store/useStore";
import { CardRenderer } from "../cards/CardRenderer";
import { AssistantBlock } from "./AssistantBlock";
import { ApprovalCard, Notice, PlanCard, UserBubble } from "./Blocks";
import { ToolCard } from "./ToolCard";

/** Consecutive items of one non-main agent are visually grouped (team workers). */
function groupItems(items: TimelineItem[]) {
  const groups: { agent: string; items: TimelineItem[] }[] = [];
  for (const it of items) {
    const agent = it.kind === "assistant" || it.kind === "tool" || it.kind === "card" ? it.agent : "main";
    const last = groups[groups.length - 1];
    if (last && last.agent === agent && agent !== "main") last.items.push(it);
    else groups.push({ agent, items: [it] });
  }
  return groups;
}

export function Timeline({ items, running }: { items: TimelineItem[]; running: boolean }) {
  const roles = useStore((s) => s.config?.roles ?? []);
  const ref = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  const dragging = useRef(false);
  const groups = useMemo(() => groupItems(items), [items]);

  const inner = useRef<HTMLDivElement>(null);
  // Follow the stream while the user is at the bottom; charts/diagrams that load later also resize the content.
  useEffect(() => {
    const el = ref.current, content = inner.current;
    if (!el || !content) return;
    const follow = () => { if (stick.current) el.scrollTop = el.scrollHeight; };
    const ro = new ResizeObserver(follow);
    ro.observe(content);
    follow();
    return () => ro.disconnect();
  }, []);

  const render = (it: TimelineItem) => {
    switch (it.kind) {
      case "user": return <UserBubble key={it.id} content={it.content} />;
      case "assistant": return <AssistantBlock key={it.id} a={it} roleColor={roles.find((r) => it.agent.startsWith(r.name))?.color} />;
      case "tool":
        // visual-card tools are represented by the card itself (unless they failed)
        if (it.name.startsWith("show_") && it.status === "done") return null;
        return <ToolCard key={it.id} t={it} />;
      case "card": return <CardRenderer key={it.id} card={it.card} agent={it.agent} />;
      case "notice": return <Notice key={it.id} level={it.level} text={it.text} />;
      case "approval": return <ApprovalCard key={it.id} it={it} />;
      case "plan": return <PlanCard key={it.id} it={it} />;
    }
  };

  return (
    <div className="timeline" ref={ref} onWheel={(e) => { if (e.deltaY < 0) stick.current = false; }}
      onTouchMove={() => { stick.current = false; }}
      onKeyDown={(e) => { if (["PageUp", "ArrowUp", "Home"].includes(e.key)) stick.current = false; }}
      onPointerDown={(e) => { dragging.current = e.target === e.currentTarget; }}
      onPointerUp={() => { dragging.current = false; }}
      onScroll={(e) => {
        // Layout changes also fire scroll events, so only explicit user intent may detach us from the bottom.
        const el = e.currentTarget;
        const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
        if (nearBottom) stick.current = true;
        else if (dragging.current) stick.current = false;
      }}>
      <div className="timeline-inner" ref={inner}>
        {groups.map((g, i) =>
          g.agent === "main" ? g.items.map(render) : (
            <div key={i} className="agent-group" style={{ borderColor: roles.find((r) => g.agent.startsWith(r.name))?.color ?? "#334155" }}>
              {g.items.map(render)}
            </div>
          ),
        )}
        {running && <div className="typing"><span /><span /><span /></div>}
      </div>
    </div>
  );
}
