import { Dot } from "@/design-system/indicators/Dot"

export type Severity = { status: "ok" | "warn" | "alert"; word: string }

/**
 * A severity as a square dot plus its word, never as a color alone. `warn` and
 * `alert` share the danger dot and differ in their ink, because the dot answers
 * "is anything wrong here" and the word answers "how much".
 *
 * `pulse` is the caller's to give, for a verdict about now: a KPI over a window
 * the reader can move into the past is a verdict about then, and stays still.
 */
export function SeverityMark({
  severity,
  pulse = false,
}: {
  severity: Severity
  pulse?: boolean
}) {
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
      <Dot
        pulse={pulse}
        className={status === "ok" ? "bg-success" : "bg-danger"}
      />
      {word.toUpperCase()}
    </span>
  )
}
