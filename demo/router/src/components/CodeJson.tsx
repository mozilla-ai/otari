import type { ReactNode } from "react";

// Pretty-printed, lightly syntax-highlighted JSON. Keys are muted, strings sit
// at body weight, numbers take the accent, and booleans/null carry status color
// so the demo's request/response boxes read like a real console, not ad-hoc divs.
function highlight(json: string): ReactNode[] {
  const token = /("(?:\\.|[^"\\])*")(\s*:)?|\b(true|false|null)\b|(-?\d+(?:\.\d+)?)/g;
  const out: ReactNode[] = [];
  let last = 0;
  let key = 0;
  for (let m = token.exec(json); m !== null; m = token.exec(json)) {
    if (m.index > last) out.push(json.slice(last, m.index));
    if (m[1] !== undefined) {
      if (m[2] !== undefined) {
        // a "key":
        out.push(
          <span key={key++} className="text-muted">
            {m[1]}
          </span>,
          m[2],
        );
      } else {
        out.push(
          <span key={key++} className="text-foreground/90">
            {m[1]}
          </span>,
        );
      }
    } else if (m[3] !== undefined) {
      const tone = m[3] === "true" ? "text-success" : m[3] === "false" ? "text-danger" : "text-muted";
      out.push(
        <span key={key++} className={tone}>
          {m[3]}
        </span>,
      );
    } else if (m[4] !== undefined) {
      out.push(
        <span key={key++} className="text-accent">
          {m[4]}
        </span>,
      );
    }
    last = token.lastIndex;
  }
  if (last < json.length) out.push(json.slice(last));
  return out;
}

export function JsonBlock({ value, className }: { value: unknown; className?: string }) {
  return (
    <pre className={["overflow-x-auto whitespace-pre-wrap break-words font-mono leading-relaxed", className].filter(Boolean).join(" ")}>
      {highlight(JSON.stringify(value, null, 2))}
    </pre>
  );
}
