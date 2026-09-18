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
  return (
    <span className="flex flex-col leading-tight">
      <span className="text-foreground">{entry.policy_name}</span>
      {/* Non-color signal: the outcome is spelled out, so the amber row tint is
          never the only thing carrying the meaning. */}
      {/* Wraps rather than truncates: the tail is the part that matters (it names
          the model that served), and a qualified target is routinely longer than
          the column. The Model column wraps for the same reason. */}
      {sentence ? (
        <span className="text-caption break-words">{sentence}</span>
      ) : null}
    </span>
  )
}
