import { formatCost, formatPct } from "@/shared/helpers/format"

import { plural } from "./format"
import type { ToolCostBreakdown, ToolCostRow } from "./toolCostBreakdown"

const ROW =
  "grid w-full grid-cols-[1fr_auto_3rem] items-baseline gap-3 px-2 py-1.5 text-left"

function Row({
  row,
  total,
  onSelect,
}: {
  row: ToolCostRow
  total: number
  onSelect: (spanId: string) => void
}) {
  return (
    <button
      type="button"
      onClick={() => onSelect(row.spanId)}
      aria-label={`${row.tool}, ${plural(row.calls, "call")}, about ${formatCost(row.cost)}${
        total > 0 ? `, ${formatPct(row.cost / total)} of the session` : ""
      }`}
      className={`${ROW} outline-none hover:bg-surface-alt focus-visible:otari-focus-ring`}
    >
      <span className="min-w-0 break-words text-body">
        {row.tool}{" "}
        <span className="text-caption text-subtle">
          {plural(row.calls, "call")}
        </span>
      </span>
      <span className="text-mono-caption">≈ {formatCost(row.cost)}</span>
      <span className="text-right text-mono-caption text-subtle">
        {total > 0 ? formatPct(row.cost / total) : null}
      </span>
    </button>
  )
}

function Plain({
  label,
  hint,
  cost,
  total,
}: {
  label: string
  hint: string
  cost: number
  total: number
}) {
  return (
    <div className={ROW}>
      <span className="min-w-0 text-body">
        {label} <span className="text-caption text-subtle">{hint}</span>
      </span>
      <span className="text-mono-caption">≈ {formatCost(cost)}</span>
      <span className="text-right text-mono-caption text-subtle">
        {total > 0 ? formatPct(cost / total) : null}
      </span>
    </div>
  )
}

// Where a session's money went, by tool, so a dear tool or a dear subagent model
// shows at a glance. Each row opens the dearest of its calls.
export function CostByTool({
  breakdown,
  total,
  onSelect,
}: {
  breakdown: ToolCostBreakdown
  total: number
  onSelect: (spanId: string) => void
}) {
  return (
    <section className="flex flex-col gap-1.5">
      <h3 className="text-overline">Cost by tool</h3>
      <p className="text-caption text-subtle">
        Estimated from the model requests around each call; what a subagent cost
        is exact. A subagent's calls are listed under the call that ran it, and
        counted there once.
      </p>
      <ul className="flex flex-col border-t border-border">
        {breakdown.rows.map((row) => (
          <li key={row.tool} className="border-b border-border">
            <Row row={row} total={total} onSelect={onSelect} />
            {row.models.length > 0 ? (
              <p className="px-2 pb-1 text-caption text-subtle">
                Its subagents ran on {row.models.join(", ")}.
              </p>
            ) : null}
            {row.inside.length > 0 ? (
              <ul
                aria-label={`Calls ${row.tool}'s subagents made`}
                className="flex flex-col pb-1 pl-4"
              >
                {row.inside.map((inner) => (
                  <li key={inner.tool}>
                    <Row row={inner} total={total} onSelect={onSelect} />
                  </li>
                ))}
                {row.ownRequests > 0 ? (
                  <li>
                    <Plain
                      label="Their own requests"
                      hint="reasoning and answers"
                      cost={row.ownRequests}
                      total={total}
                    />
                  </li>
                ) : null}
              </ul>
            ) : null}
          </li>
        ))}
        {breakdown.judges > 0 ? (
          <li className="border-b border-border">
            <Plain
              label="Judge gates"
              hint="their model calls"
              cost={breakdown.judges}
              total={total}
            />
          </li>
        ) : null}
        {breakdown.untied > 0 ? (
          <li className="border-b border-border">
            <Plain
              label="No tool call"
              hint="prompts, the system prompt and answers"
              cost={breakdown.untied}
              total={total}
            />
          </li>
        ) : null}
      </ul>
    </section>
  )
}
