import { useState } from "react";
import { Alert, Button, Descriptions, Progress, Tag } from "antd";
import { CheckCircleFilled, CloseCircleFilled } from "@ant-design/icons";
import { api } from "../../api/client";
import { fmtTokens } from "../../lib/format";
import { useActiveSession, useChat, useStore } from "../../store/useStore";

export function ContextPanel() {
  const config = useStore((s) => s.config)!;
  const health = useStore((s) => s.health);
  const session = useActiveSession()!;
  const chat = useChat();
  const [test, setTest] = useState<Awaited<ReturnType<typeof api.selftest>> | null>(null);
  const [busy, setBusy] = useState(false);
  const budget = config.model.context_budget;
  const u = { ...session.usage, ...chat.usage };
  return (
    <div className="panel-pad">
      <h4>Context window</h4>
      <Progress percent={Math.min(100, Math.round((chat.context.tokens / budget) * 100))} format={() => `${fmtTokens(chat.context.tokens)} / ${fmtTokens(budget)}`} />
      <p className="muted small">The model was tuned on short agent traces, so history is summarised well before the {fmtTokens(config.model.num_ctx)}-token window is full.</p>
      <h4>Usage (this session)</h4>
      <Descriptions size="small" column={2} colon={false}>
        <Descriptions.Item label="Input tokens">{fmtTokens(u.input_tokens ?? 0)}</Descriptions.Item>
        <Descriptions.Item label="Output tokens">{fmtTokens(u.output_tokens ?? 0)}</Descriptions.Item>
        <Descriptions.Item label="LLM calls">{u.llm_calls ?? 0}</Descriptions.Item>
        <Descriptions.Item label="Tool calls">{u.tool_calls ?? 0}</Descriptions.Item>
      </Descriptions>
      <h4>Model</h4>
      <Descriptions size="small" column={1} colon={false}>
        <Descriptions.Item label="Tag"><span className="mono">{config.model.tag}</span></Descriptions.Item>
        <Descriptions.Item label="Architecture">{config.model.family}</Descriptions.Item>
        <Descriptions.Item label="Serving">{config.model.backend} · raw completion with the model's native chat template</Descriptions.Item>
        <Descriptions.Item label="Thinking budget">{config.model.thinking_budget} tokens</Descriptions.Item>
        <Descriptions.Item label="Status">{health?.model?.ok ? <Tag color="success" bordered={false}>reachable</Tag> : <Tag color="error" bordered={false}>unreachable</Tag>}</Descriptions.Item>
      </Descriptions>
      <h4>Sandbox</h4>
      <Descriptions size="small" column={1} colon={false}>
        <Descriptions.Item label="Backend">{config.sandbox.backend}</Descriptions.Item>
        <Descriptions.Item label="Image"><span className="mono">{config.sandbox.image}</span></Descriptions.Item>
        <Descriptions.Item label="Limits">{config.sandbox.cpus} CPU · {config.sandbox.memory} · network {config.sandbox.network}</Descriptions.Item>
      </Descriptions>
      {config.sandbox.backend === "local" && <Alert type="warning" showIcon message="Local sandbox backend is NOT isolated. Use the docker backend in production." />}
      <h4>Model self-test</h4>
      <p className="muted small">Runs live checks of tool calling, typed arguments, thinking and structured output against the model server.</p>
      <Button size="small" loading={busy} onClick={async () => { setBusy(true); try { setTest(await api.selftest()); } finally { setBusy(false); } }}>Run self-test</Button>
      {test && (
        <div className="selftest">
          {test.results.map((r) => (
            <div key={r.name} className="st-row">{r.passed ? <CheckCircleFilled className="good" /> : <CloseCircleFilled className="bad" />}<div><b>{r.name}</b> <span className="muted small">{r.seconds}s</span><div className="muted small">{r.detail}</div></div></div>
          ))}
        </div>
      )}
    </div>
  );
}
