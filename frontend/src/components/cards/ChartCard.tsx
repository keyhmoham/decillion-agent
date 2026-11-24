import { Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, Pie, PieChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";

const PALETTE = ["#818cf8", "#34d399", "#fbbf24", "#f87171", "#38bdf8", "#c084fc", "#fb923c", "#2dd4bf", "#f472b6", "#a3e635"];

interface ChartData { type: "bar" | "line" | "area" | "pie" | "scatter"; labels: string[]; series: { name: string; values: number[] }[]; unit?: string | null }

export default function ChartCard({ data }: { data: ChartData }) {
  const rows = data.labels.map((l, i) => ({ label: l, ...Object.fromEntries(data.series.map((s) => [s.name, s.values[i]])) }));
  const unit = data.unit ?? "";
  const axis = { stroke: "#64748b", fontSize: 12, tickLine: false } as const;
  const tip = {
    contentStyle: { background: "#111827", border: "1px solid #334155", borderRadius: 8, fontSize: 12 },
    formatter: (v: any) => `${v}${unit ? " " + unit : ""}`,
  };
  const grid = <CartesianGrid stroke="#1f2937" strokeDasharray="3 3" vertical={false} />;
  const legend = data.series.length > 1 ? <Legend wrapperStyle={{ fontSize: 12 }} /> : null;
  let chart: React.ReactElement;
  switch (data.type) {
    case "line":
      chart = <LineChart data={rows}>{grid}<XAxis dataKey="label" {...axis} /><YAxis {...axis} width={48} /><Tooltip {...tip} />{legend}{data.series.map((s, i) => <Line key={s.name} type="monotone" dataKey={s.name} stroke={PALETTE[i % PALETTE.length]} strokeWidth={2} dot={rows.length < 25} />)}</LineChart>;
      break;
    case "area":
      chart = <AreaChart data={rows}>{grid}<XAxis dataKey="label" {...axis} /><YAxis {...axis} width={48} /><Tooltip {...tip} />{legend}{data.series.map((s, i) => <Area key={s.name} type="monotone" dataKey={s.name} stroke={PALETTE[i % PALETTE.length]} fill={PALETTE[i % PALETTE.length]} fillOpacity={0.25} />)}</AreaChart>;
      break;
    case "pie": {
      const s = data.series[0];
      const slices = data.labels.map((l, i) => ({ name: l, value: s.values[i] }));
      chart = <PieChart><Tooltip {...tip} /><Legend wrapperStyle={{ fontSize: 12 }} /><Pie data={slices} dataKey="value" nameKey="name" innerRadius="45%" outerRadius="80%" paddingAngle={2} stroke="none">{slices.map((_, i) => <Cell key={i} fill={PALETTE[i % PALETTE.length]} />)}</Pie></PieChart>;
      break;
    }
    case "scatter": {
      const pts = data.series.map((s) => ({ name: s.name, data: s.values.map((v, i) => ({ x: i, y: v, label: data.labels[i] })) }));
      chart = <ScatterChart>{grid}<XAxis type="number" dataKey="x" {...axis} tickFormatter={(i) => data.labels[i] ?? i} /><YAxis type="number" dataKey="y" {...axis} width={48} /><Tooltip {...tip} />{legend}{pts.map((p, i) => <Scatter key={p.name} name={p.name} data={p.data} fill={PALETTE[i % PALETTE.length]} />)}</ScatterChart>;
      break;
    }
    default:
      chart = <BarChart data={rows}>{grid}<XAxis dataKey="label" {...axis} /><YAxis {...axis} width={48} /><Tooltip {...tip} cursor={{ fill: "#1f293755" }} />{legend}{data.series.map((s, i) => <Bar key={s.name} dataKey={s.name} fill={PALETTE[i % PALETTE.length]} radius={[4, 4, 0, 0]} />)}</BarChart>;
  }
  return <div style={{ width: "100%", height: 280 }}><ResponsiveContainer>{chart}</ResponsiveContainer></div>;
}
