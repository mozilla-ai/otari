import type { TraceGateSummary } from "@/client"
import { Chip, type ChipTone } from "@/design-system/indicators/Chip"

const GATE_COUNTS: {
  key: "blocked" | "refused" | "undecided" | "flagged" | "passed"
  label: string
  tone: ChipTone
}[] = [
  { key: "blocked", label: "blocked", tone: "warning" },
  { key: "refused", label: "could not decide, held", tone: "danger" },
  { key: "undecided", label: "could not decide", tone: "danger" },
  { key: "flagged", label: "flagged", tone: "warning" },
  { key: "passed", label: "passed", tone: "success" },
]

function checks(count: number): string {
  return count === 1 ? "1 check" : `${count} checks`
}

// A gate that could not decide but is required holds the agent as surely as one that blocked.
export function stoppedBy(gate: TraceGateSummary): number {
  return gate.blocked + gate.refused
}

function total(gate: TraceGateSummary): number {
  return (
    gate.passed + gate.blocked + gate.flagged + gate.refused + gate.undecided
  )
}

// Every gate the agent's own hook ran, passes included, then the ones its
// guardrail declares that never ran here. Each row that ran opens the gate's
// first check that did not pass.
export function GatesView({
  gates,
  onSelect,
}: {
  gates: TraceGateSummary[]
  onSelect: (spanId: string) => void
}) {
  const ran = gates.filter((gate) => total(gate) > 0)
  const idle = gates.filter((gate) => total(gate) === 0)
  const count = ran.reduce((sum, gate) => sum + total(gate), 0)
  const blocked = ran.reduce((sum, gate) => sum + stoppedBy(gate), 0)
  return (
    <div className="flex w-full flex-col gap-4 p-4">
      <section className="flex flex-col gap-1.5">
        <h3 className="text-overline">Agent guardrail gates</h3>
        <p className="text-caption text-subtle">
          {ran.length === 1 ? "1 gate" : `${ran.length} gates`} ran{" "}
          {checks(count)}
          {blocked > 0
            ? `, and stopped the agent ${blocked === 1 ? "once" : `${blocked} times`}.`
            : ", and none stopped the agent."}
        </p>
        <ul className="flex flex-col border-t border-border">
          {ran.map((gate) => (
            <li key={gate.gate_id} className="border-b border-border">
              <button
                type="button"
                disabled={!gate.span_id}
                onClick={() => gate.span_id && onSelect(gate.span_id)}
                className="flex min-h-11 w-full flex-col gap-1 px-2 py-2 text-left outline-none enabled:hover:bg-surface-alt focus-visible:otari-focus-ring"
              >
                <span className="flex w-full flex-wrap items-center gap-x-3 gap-y-1">
                  <span className="min-w-0 flex-1 break-words text-mono-caption">
                    {gate.gate_id}
                  </span>
                  <span className="flex flex-wrap gap-1">
                    {GATE_COUNTS.filter(({ key }) => gate[key] > 0).map(
                      ({ key, label, tone }) => (
                        <Chip key={key} tone={tone}>
                          {gate[key]} {label}
                        </Chip>
                      ),
                    )}
                  </span>
                </span>
                {gate.message ? (
                  <span className="text-caption text-subtle">
                    {gate.message}
                  </span>
                ) : null}
              </button>
            </li>
          ))}
        </ul>
      </section>
      {idle.length > 0 ? (
        <section className="flex flex-col gap-1.5">
          <h3 className="text-overline">Did not run in this session</h3>
          <p className="text-caption text-subtle">
            The guardrail declares these, and nothing this session did reached
            them. A gate that never runs in any session may be scoped wrong.
          </p>
          <ul className="flex flex-col border-t border-border">
            {idle.map((gate) => (
              <li
                key={gate.gate_id}
                className="flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-border px-2 py-2"
              >
                <span className="min-w-0 flex-1 break-words text-mono-caption text-subtle">
                  {gate.gate_id}
                </span>
                {gate.gate_type ? (
                  <Chip tone="neutral">{gate.gate_type}</Chip>
                ) : null}
              </li>
            ))}
          </ul>
        </section>
      ) : null}
    </div>
  )
}
