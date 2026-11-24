import { Tooltip } from "antd";
import { ApartmentOutlined, CodeOutlined, DashboardOutlined, DatabaseOutlined, FolderOpenOutlined, ToolOutlined } from "@ant-design/icons";
import { useStore, PanelTab } from "../store/useStore";
import { ContextPanel } from "./panels/ContextPanel";
import { FilesPanel } from "./panels/FilesPanel";
import { MemoryPanel } from "./panels/MemoryPanel";
import { PlanPanel } from "./panels/PlanPanel";
import { TerminalPanel } from "./panels/TerminalPanel";
import { ToolsPanel } from "./panels/ToolsPanel";

const TABS: { key: PanelTab; label: string; icon: React.ReactNode }[] = [
  { key: "files", label: "Files", icon: <FolderOpenOutlined /> },
  { key: "terminal", label: "Terminal", icon: <CodeOutlined /> },
  { key: "plan", label: "Plan", icon: <ApartmentOutlined /> },
  { key: "memory", label: "Memory", icon: <DatabaseOutlined /> },
  { key: "tools", label: "Tools", icon: <ToolOutlined /> },
  { key: "context", label: "Context", icon: <DashboardOutlined /> },
];

export function RightPanel() {
  const tab = useStore((s) => s.panelTab);
  const setPanel = useStore((s) => s.setPanel);
  return (
    <aside className="rightpanel">
      <div className="ptabs" role="tablist">
        {TABS.map((t) => (
          <Tooltip key={t.key} title={t.label} placement="bottom">
            <button role="tab" aria-selected={tab === t.key} className={`ptab ${tab === t.key ? "on" : ""}`} onClick={() => setPanel(true, t.key)}>
              {t.icon}{tab === t.key && <span>{t.label}</span>}
            </button>
          </Tooltip>
        ))}
      </div>
      <div className="panel-body">
        {tab === "files" && <FilesPanel />}
        {tab === "terminal" && <TerminalPanel />}
        {tab === "plan" && <PlanPanel />}
        {tab === "memory" && <MemoryPanel />}
        {tab === "tools" && <ToolsPanel />}
        {tab === "context" && <ContextPanel />}
      </div>
    </aside>
  );
}
