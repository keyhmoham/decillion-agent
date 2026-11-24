import { useEffect, useState } from "react";
import { Alert, Button, ConfigProvider, Input, Result, Spin, theme } from "antd";
import { setApiKey } from "./api/client";
import { ChatView } from "./components/chat/ChatView";
import { RightPanel } from "./components/RightPanel";
import { Sidebar } from "./components/Sidebar";
import { useStore } from "./store/useStore";

function Shell() {
  const panelOpen = useStore((s) => s.panelOpen);
  const activeId = useStore((s) => s.activeId);
  return (
    <div className={`app ${panelOpen && activeId ? "with-panel" : ""}`}>
      <Sidebar />
      <main className="main"><ChatView /></main>
      {panelOpen && activeId && <RightPanel />}
    </div>
  );
}

function AuthGate() {
  const [key, setKey] = useState("");
  const boot = useStore((s) => s.boot);
  return (
    <div className="center-screen">
      <Result status="info" title="API key required" subTitle="This deployment is protected. Enter the APP_API_KEY configured for the backend."
        extra={<div className="auth-form"><Input.Password value={key} onChange={(e) => setKey(e.target.value)} onPressEnter={() => { setApiKey(key); boot(); }} placeholder="API key" /><Button type="primary" onClick={() => { setApiKey(key); boot(); }}>Continue</Button></div>} />
    </div>
  );
}

export default function App() {
  const boot = useStore((s) => s.boot);
  const booting = useStore((s) => s.booting);
  const fatal = useStore((s) => s.fatal);
  useEffect(() => { boot(); }, [boot]);
  return (
    <ConfigProvider
      theme={{
        algorithm: theme.darkAlgorithm,
        token: { colorPrimary: "#6366f1", borderRadius: 8, colorBgBase: "#0b0f17", colorBgContainer: "#111827", colorBgElevated: "#151c2b", fontFamily: "Inter, ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif", fontSize: 14 },
        components: { Table: { headerBg: "#0f1623", rowHoverBg: "#162033" }, Tabs: { horizontalMargin: "0 0 0 0" } },
      }}
    >
      {booting ? <div className="center-screen"><Spin size="large" /></div> : fatal === "auth" ? <AuthGate /> : fatal ? (
        <div className="center-screen"><Alert type="error" showIcon message="Cannot reach the backend" description={fatal} action={<Button onClick={() => boot()}>Retry</Button>} /></div>
      ) : <Shell />}
    </ConfigProvider>
  );
}
