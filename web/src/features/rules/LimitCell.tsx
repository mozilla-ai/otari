import { accessWords, type Rule } from "@/features/rules/rules"
import { formatUsd } from "@/shared/helpers/format"

/** The limit: an amount and how often it starts over, or an allow-list's size. */
export function LimitCell({ rule }: { rule: Rule }) {
  if (rule.spend) {
    const per = [rule.spend.isEach ? "each" : "", rule.spend.per]
      .filter(Boolean)
      .join(", ")
    return (
      <span className="flex min-w-0 items-center gap-2">
        <span className="shrink-0 text-mono-caption text-foreground tabular-nums">
          {formatUsd(rule.spend.max)}
        </span>
        {per ? (
          <span className="min-w-0 truncate text-sm text-muted">{per}</span>
        ) : null}
      </span>
    )
  }
  if (rule.access) {
    return (
      <span className="text-sm text-muted">{accessWords(rule.access)}</span>
    )
  }
  return null
}
