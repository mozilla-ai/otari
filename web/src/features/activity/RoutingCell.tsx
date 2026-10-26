import type { UsageEntry } from "@/client"
import { describeAttempt, type GroupOutcome } from "./activityModel"

// The Routing column: the policy the caller named, then where this row sits in its
// plan and how that turned out.
export function RoutingCell({
  entry,
  outcome,
}: {
  entry: UsageEntry
  outcome: GroupOutcome | null
}) {
  // Blank, not an em-dash, when the request named a plain model. This column is
  // sparse by nature (most rows are unrouted), and a placeholder on every one of
  // them would add noise to every scan while saying nothing.
  if (entry.policy_name == null) return null
  const sentence = describeAttempt(entry, outcome)
  // One line in the table, from md up: the policy keeps its width first and the
  // outcome takes what is left, since Status already says how the attempt went.
  // The title holds the whole of it. A phone's card has no row pitch to keep, so
  // it stacks and wraps.
  return (
    <span
      className="flex flex-col leading-tight md:flex-row md:items-baseline md:gap-2"
      title={sentence ? `${entry.policy_name}: ${sentence}` : entry.policy_name}
    >
      <span className="text-foreground md:max-w-full md:shrink-0 md:truncate">
        {entry.policy_name}
      </span>
      {/* Non-color signal: the outcome is spelled out, so the amber row tint is
          never the only thing carrying the meaning. */}
      {sentence ? (
        <span className="text-caption break-words md:min-w-0 md:truncate">
          {sentence}
        </span>
      ) : null}
    </span>
  )
}
