import { Button, Dropdown, Modal, Tooltip } from "antd";
import { DeleteOutlined, MoreOutlined, PlusOutlined, MessageOutlined, LoadingOutlined } from "@ant-design/icons";
import { useStore } from "../store/useStore";
import { relTime } from "../lib/format";

export function Sidebar() {
  const sessions = useStore((s) => s.sessions);
  const activeId = useStore((s) => s.activeId);
  const chats = useStore((s) => s.chats);
  const select = useStore((s) => s.select);
  const create = useStore((s) => s.newSession);
  const remove = useStore((s) => s.removeSession);
  const config = useStore((s) => s.config);
  return (
    <aside className="sidebar">
      <div className="brand"><div className="logo">D</div><div><b>Decillion</b><span>Agent</span></div></div>
      <Button type="primary" block icon={<PlusOutlined />} onClick={() => create()} className="new-btn">New session</Button>
      <div className="session-list" role="list">
        {sessions.map((s) => (
          <div key={s.id} role="listitem" className={`session-item ${s.id === activeId ? "active" : ""}`} onClick={() => select(s.id)}>
            {chats[s.id]?.running ? <LoadingOutlined spin /> : <MessageOutlined />}
            <div className="session-text">
              <div className="session-title">{s.title}</div>
              <div className="session-sub">{relTime(s.updated_at)} · {s.config.mode}</div>
            </div>
            <Dropdown
              trigger={["click"]}
              menu={{ items: [{ key: "del", icon: <DeleteOutlined />, danger: true, label: "Delete session & sandbox", onClick: (i) => { i.domEvent.stopPropagation(); Modal.confirm({ title: "Delete this session?", content: "Its Docker sandbox and all workspace files will be removed.", okButtonProps: { danger: true }, okText: "Delete", onOk: () => remove(s.id) }); } }] }}
            >
              <Button type="text" size="small" icon={<MoreOutlined />} onClick={(e) => e.stopPropagation()} aria-label="Session menu" />
            </Dropdown>
          </div>
        ))}
        {!sessions.length && <div className="muted pad">No sessions yet.</div>}
      </div>
      <div className="sidebar-foot">
        <Tooltip title={`${config?.model.family}`}><span className="muted">{config?.model.tag.replace("nutboy02/", "")}</span></Tooltip>
        <span className="muted">v{config?.version}</span>
      </div>
    </aside>
  );
}
