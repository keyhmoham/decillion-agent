import { useEffect, useRef, useState } from "react";
import { api } from "../../api/client";
import { stripAnsi } from "../../lib/format";
import { useStore } from "../../store/useStore";

interface Line { kind: "cmd" | "out" | "err" | "meta"; text: string }

export function TerminalPanel() {
  const sid = useStore((s) => s.activeId)!;
  const [lines, setLines] = useState<Line[]>([{ kind: "meta", text: "Interactive shell in this session's sandbox. Commands run as the agent user." }]);
  const [cmd, setCmd] = useState("");
  const [busy, setBusy] = useState(false);
  const [hist, setHist] = useState<string[]>([]);
  const [hi, setHi] = useState(-1);
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => { setLines([{ kind: "meta", text: "Interactive shell in this session's sandbox. Commands run as the agent user." }]); }, [sid]);
  useEffect(() => end.current?.scrollIntoView({ block: "end" }), [lines]);

  const run = async () => {
    const c = cmd.trim();
    if (!c || busy) return;
    setCmd(""); setHist((h) => [...h, c]); setHi(-1); setBusy(true);
    setLines((l) => [...l, { kind: "cmd", text: c }]);
    try {
      const r = await api.terminal(sid, c, 120);
      setLines((l) => [...l, ...(r.stdout ? [{ kind: "out" as const, text: stripAnsi(r.stdout) }] : []), ...(r.stderr ? [{ kind: "err" as const, text: stripAnsi(r.stderr) }] : []),
        { kind: "meta", text: r.timed_out ? "timed out" : `exit ${r.exit_code} · ${r.duration}s` }]);
    } catch (e: any) { setLines((l) => [...l, { kind: "err", text: e.message }]); }
    setBusy(false);
  };
  return (
    <div className="terminal">
      <div className="term-scroll">
        {lines.map((l, i) => <pre key={i} className={`tl tl-${l.kind}`}>{l.kind === "cmd" ? `$ ${l.text}` : l.text}</pre>)}
        <div ref={end} />
      </div>
      <div className="term-input">
        <span>$</span>
        <input
          value={cmd} disabled={busy} autoFocus spellCheck={false} placeholder={busy ? "running…" : "type a command"}
          onChange={(e) => setCmd(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") run();
            else if (e.key === "ArrowUp" && hist.length) { const n = hi < 0 ? hist.length - 1 : Math.max(0, hi - 1); setHi(n); setCmd(hist[n]); }
            else if (e.key === "ArrowDown" && hi >= 0) { const n = hi + 1; if (n >= hist.length) { setHi(-1); setCmd(""); } else { setHi(n); setCmd(hist[n]); } }
          }}
        />
      </div>
    </div>
  );
}
