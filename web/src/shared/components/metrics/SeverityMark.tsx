import { Dot } from "@/shared/components/indicators/Dot"

export type Severity = { status: "ok" | "warn" | "alert"; word: string }

/**
 * A severity as a square dot plus its word, never as a color alone. `warn` and
 * `alert` share the danger dot and differ in their ink, because the dot answers
 * "is anything wrong here" and the word answers "how much".
 */
export function SeverityMark({ severity }: { severity: Severity }) {
  const { status, word } = severity
  return (
    <span
      className={`flex shrink-0 items-center gap-2 font-mono ${
        status === "alert"
          ? "text-danger"
          : status === "warn"
            ? "text-muted"
            : "text-subtle"
      }`}
    >
      <Dot className={status === "ok" ? "bg-success" : "bg-danger"} />
      {word.toUpperCase()}
    </span>
  )
}
