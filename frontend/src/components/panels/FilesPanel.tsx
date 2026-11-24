import { useCallback, useEffect, useState } from "react";
import { Button, Empty, Input, message, Spin, Tooltip } from "antd";
import { DownloadOutlined, FileOutlined, FolderFilled, FolderOpenFilled, ReloadOutlined, SaveOutlined, ArrowLeftOutlined } from "@ant-design/icons";
import hljs from "highlight.js/lib/common";
import { api, downloadUrl, fileUrl } from "../../api/client";
import type { FileEntry } from "../../api/types";
import { baseName, extToLang } from "../../lib/format";
import { useChat, useStore } from "../../store/useStore";

function Node({ sid, entry, depth, onOpen, dirty }: { sid: string; entry: FileEntry; depth: number; onOpen: (p: string) => void; dirty: number }) {
  const [open, setOpen] = useState(false);
  const [kids, setKids] = useState<FileEntry[] | null>(null);
  useEffect(() => {
    if (open) api.files(sid, entry.path).then((r) => setKids(r.entries)).catch(() => setKids([]));
  }, [open, dirty, sid, entry.path]);
  return (
    <>
      <div className="fnode" style={{ paddingLeft: 8 + depth * 14 }} onClick={() => (entry.is_dir ? setOpen(!open) : onOpen(entry.path))} role="treeitem">
        {entry.is_dir ? (open ? <FolderOpenFilled className="folder" /> : <FolderFilled className="folder" />) : <FileOutlined />}
        <span>{entry.name}</span>
        {!entry.is_dir && <span className="fsize">{entry.size < 1024 ? `${entry.size} B` : `${(entry.size / 1024).toFixed(1)} KB`}</span>}
      </div>
      {open && kids?.map((k) => <Node key={k.path} sid={sid} entry={k} depth={depth + 1} onOpen={onOpen} dirty={dirty} />)}
    </>
  );
}

function Viewer({ sid, path, onBack }: { sid: string; path: string; onBack: () => void }) {
  const [data, setData] = useState<{ content: string; binary: boolean; mime: string } | null>(null);
  const [draft, setDraft] = useState<string | null>(null);
  const isImage = /\.(png|jpe?g|gif|svg|webp)$/i.test(path);
  useEffect(() => {
    setData(null); setDraft(null);
    if (!isImage) api.file(sid, path).then(setData).catch((e) => message.error(e.message));
  }, [sid, path, isImage]);
  const lang = extToLang(path);
  const html = data && !data.binary && draft === null ? (() => { try { return lang && hljs.getLanguage(lang) ? hljs.highlight(data.content, { language: lang }).value : hljs.highlightAuto(data.content.slice(0, 50000)).value; } catch { return ""; } })() : "";
  return (
    <div className="viewer">
      <div className="viewer-head">
        <Button type="text" size="small" icon={<ArrowLeftOutlined />} onClick={onBack} />
        <span className="viewer-path" title={path}>{baseName(path)}</span>
        {draft !== null && <Button type="primary" size="small" icon={<SaveOutlined />} onClick={async () => { await api.writeFile(sid, path, draft); message.success("Saved"); setData((d) => d && { ...d, content: draft }); setDraft(null); }}>Save</Button>}
      </div>
      {isImage ? <div className="viewer-img"><img src={fileUrl(sid, path)} alt={path} /></div> : !data ? <Spin className="pad" /> : data.binary ? <Empty description="Binary file" /> : draft !== null ? (
        <textarea className="editor" value={draft} onChange={(e) => setDraft(e.target.value)} spellCheck={false} />
      ) : (
        <pre className="viewer-code hljs" onDoubleClick={() => setDraft(data.content)} title="Double-click to edit"><code dangerouslySetInnerHTML={{ __html: html }} /></pre>
      )}
    </div>
  );
}

export function FilesPanel() {
  const sid = useStore((s) => s.activeId)!;
  const chat = useChat();
  const [entries, setEntries] = useState<FileEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [file, setFile] = useState<string | null>(null);
  const [workdir, setWorkdir] = useState("");
  const [filter, setFilter] = useState("");
  const load = useCallback(() => {
    setLoading(true);
    api.files(sid).then((r) => { setEntries(r.entries); setWorkdir(r.workdir); }).catch(() => setEntries([])).finally(() => setLoading(false));
  }, [sid]);
  useEffect(() => { setFile(null); }, [sid]);
  useEffect(load, [load, chat.dirtyFiles]);
  if (file) return <Viewer sid={sid} path={file} onBack={() => setFile(null)} />;
  const shown = entries.filter((e) => e.name.toLowerCase().includes(filter.toLowerCase()));
  return (
    <div className="panel-pad">
      <div className="panel-toolbar">
        <Input size="small" placeholder="Filter…" allowClear value={filter} onChange={(e) => setFilter(e.target.value)} />
        <Tooltip title="Refresh"><Button size="small" type="text" icon={<ReloadOutlined spin={loading} />} onClick={load} /></Tooltip>
        <Tooltip title="Download workspace (.zip)"><Button size="small" type="text" icon={<DownloadOutlined />} href={downloadUrl(sid)} /></Tooltip>
      </div>
      <div className="muted mono small">{workdir}</div>
      <div className="ftree" role="tree">
        {shown.map((e) => <Node key={e.path} sid={sid} entry={e} depth={0} onOpen={setFile} dirty={chat.dirtyFiles} />)}
        {!shown.length && !loading && <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="Workspace is empty" />}
      </div>
    </div>
  );
}
