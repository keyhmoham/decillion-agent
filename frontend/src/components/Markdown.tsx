import { memo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import { CopyOutlined, CheckOutlined } from "@ant-design/icons";
import { useState } from "react";

function CodeBlock({ children, className }: { children?: React.ReactNode; className?: string }) {
  const [copied, setCopied] = useState(false);
  const lang = /language-(\w+)/.exec(className ?? "")?.[1];
  const text = String(children ?? "").replace(/\n$/, "");
  const inline = !className && !text.includes("\n");
  if (inline) return <code className="inline-code">{children}</code>;
  return (
    <div className="codeblock">
      <div className="codeblock-head">
        <span>{lang ?? "text"}</span>
        <button
          className="icon-btn"
          onClick={() => { navigator.clipboard?.writeText(text); setCopied(true); setTimeout(() => setCopied(false), 1200); }}
          aria-label="Copy code"
        >
          {copied ? <CheckOutlined /> : <CopyOutlined />}
        </button>
      </div>
      <pre><code className={className}>{children}</code></pre>
    </div>
  );
}

export const Markdown = memo(function Markdown({ text }: { text: string }) {
  return (
    <div className="md">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[[rehypeHighlight, { detect: true, ignoreMissing: true }]]}
        components={{
          code: ({ className, children }) => <CodeBlock className={className}>{children}</CodeBlock>,
          pre: ({ children }) => <>{children}</>,
          a: ({ href, children }) => <a href={href} target="_blank" rel="noreferrer noopener">{children}</a>,
          table: ({ children }) => <div className="md-table"><table>{children}</table></div>,
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
});
