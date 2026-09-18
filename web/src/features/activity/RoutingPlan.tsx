import type { UsageEntry } from "@/client"
import { useRequestGroups } from "@/shared/api/usage"
import {
  describeAttemptOutcome,
  describeSelectionReason,
  findPricingSelector,
  formatLatencyCell,
  formatUSD,
  sortPlanRows,
} from "./activityModel"

// The whole plan behind one routed request: every candidate that ran, in order,
// with the one that served marked. This is the answer to "a fallback fired, so
// what actually served me", which no single row can give.
export function RoutingPlan({ entry }: { entry: UsageEntry }) {
  const groupId = entry.request_group_id
  const groupIds = groupId ? [groupId] : []
  const group = useRequestGroups(groupIds)
  // Only rows of this row's own group are the plan. The lookup keeps previous data
  // across a key change, so this is what stops another request's plan from ever
  // being narrated as this one's, whatever the detail panel does with mounting.
  const siblings = groupId
    ? (group.data ?? []).filter((row) => row.request_group_id === groupId)
    : []
  // Falls back to the row itself while the lookup is in flight (and for a
  // pre-`request_group_id` row, which has no siblings to find), so the section
  // never flashes empty and never claims a one-attempt plan it did not read.
  const attempts = sortPlanRows(siblings.length ? siblings : [entry])
  const isComplete = siblings.length > 0
  const served = attempts.find((attempt) => attempt.status === "success")
  const total = entry.attempt_count ?? attempts.length

  // "Loading" only while a lookup is actually outstanding: a failed lookup, or a row
  // that carries no group to look up, would otherwise sit on that line forever.
  const summary = !isComplete
    ? group.isError
      ? "Could not load this request's other attempts."
      : entry.request_group_id
        ? "Loading the rest of this request's attempts…"
        : "This row carries no request group, so its other attempts cannot be found."
    : served
      ? `Served by attempt ${served.attempt_position ?? "?"} of ${total}: ${findPricingSelector(served)}`
      : attempts.some((attempt) => attempt.status === "error")
        ? "No candidate served this request."
        : "This request has no outcome row yet."

  return (
    <div className="flex flex-col gap-2">
      <span className="text-overline">Routing plan · {entry.policy_name}</span>
      <span className="text-sm text-foreground">{summary}</span>
      <div className="overflow-x-auto border border-control-border">
        <table
          className="w-full text-xs"
          aria-label={`Routing plan for policy ${entry.policy_name}`}
        >
          {/* No `text-muted` here: `text-overline` on each `<th>` sets the color
              itself, so a second declaration on the parent is one more place to
              keep in step for no effect. */}
          <thead>
            <tr className="border-b border-border">
              <th scope="col" className="px-3 py-2 text-left text-overline">
                #
              </th>
              <th scope="col" className="px-3 py-2 text-left text-overline">
                Target
              </th>
              <th scope="col" className="px-3 py-2 text-left text-overline">
                Selected as
              </th>
              <th scope="col" className="px-3 py-2 text-left text-overline">
                Outcome
              </th>
              {/* Every attempt's `latency_ms` is measured from the start of the
                  request, not from the start of that attempt, so this is the same
                  "Total time" the row column shows, not a per-candidate duration. */}
              <th scope="col" className="px-3 py-2 text-right text-overline">
                Total time
              </th>
              <th scope="col" className="px-3 py-2 text-right text-overline">
                Cost
              </th>
            </tr>
          </thead>
          <tbody>
            {attempts.map((attempt) => (
              <tr
                key={attempt.id}
                className={`border-t border-border first:border-t-0 ${
                  attempt.status === "success" ? "bg-primary-subtle" : ""
                }`}
              >
                <td className="px-3 py-2 tabular-nums">
                  {attempt.attempt_position ?? "?"}
                </td>
                <td className="px-3 py-2 break-all text-foreground">
                  {findPricingSelector(attempt)}
                  {attempt.id === entry.id ? (
                    <span className="ml-2 border border-border px-1.5 py-0.5 text-xs text-subtle">
                      this row
                    </span>
                  ) : null}
                </td>
                <td className="px-3 py-2">
                  {describeSelectionReason(attempt.selection_reason) ?? "—"}
                </td>
                <td
                  className={`px-3 py-2 ${attempt.status === "success" ? "" : "text-warning"}`}
                >
                  {describeAttemptOutcome(attempt)}
                </td>
                <td className="px-3 py-2 text-right tabular-nums">
                  {formatLatencyCell(attempt.latency_ms)}
                </td>
                <td className="px-3 py-2 text-right tabular-nums">
                  {formatUSD(attempt.cost)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <span className="text-caption">
        Cost and tool charges settle on the attempt that served, so a failed
        attempt carries its tokens and no charge.
      </span>
    </div>
  )
}
