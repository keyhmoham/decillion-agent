export const fmtTokens = (n: number) => (n >= 10000 ? `${(n / 1000).toFixed(1)}k` : String(n));
export const baseName = (p: string) => p.replace(/\/+$/, "").split("/").pop() || p;
export function extToLang(path: string): string | undefined {
  const ext = path.split(".").pop()?.toLowerCase();
  const map: Record<string, string> = { py: "python", js: "javascript", ts: "typescript", tsx: "tsx", jsx: "jsx", json: "json", md: "markdown", sh: "bash", yml: "yaml", yaml: "yaml", html: "xml", css: "css", rs: "rust", go: "go", java: "java", c: "c", cpp: "cpp", h: "c", sql: "sql", toml: "ini" };
  return ext ? map[ext] : undefined;
}
export const stripAnsi = (s: string) => s.replace(/\u001b\[[0-9;?]*[ -/]*[@-~]/g, "");
export function relTime(iso: string): string {
  const d = (Date.now() - new Date(iso).getTime()) / 1000;
  if (d < 60) return "just now";
  if (d < 3600) return `${Math.floor(d / 60)}m ago`;
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`;
  return `${Math.floor(d / 86400)}d ago`;
}
