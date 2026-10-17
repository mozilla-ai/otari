import type { UsageEntry } from "@/client"
import { formatNumber } from "@/shared/helpers/format"
import { buildTokenComposition, formatTokenCount } from "./activityModel"

// Billed input, cache reads and writes included, so Input and Output add up to
// the row's billed tokens. On a cached agent call the cache is most of that
// number, so the split is the title and is read out after it; the detail panel
// has every raw field.
export function InputTokens({ entry }: { entry: UsageEntry }) {
  const composition = buildTokenComposition(entry)
  if (composition === null) return formatTokenCount(entry.prompt_tokens)
  const { fresh, cacheRead, cacheWrite } = composition
  const input = formatNumber(fresh + cacheRead + cacheWrite)
  if (!cacheRead && !cacheWrite) return input
  const split = [
    `Fresh input ${formatNumber(fresh)}`,
    cacheRead ? `cache read ${formatNumber(cacheRead)}` : null,
    cacheWrite ? `cache write ${formatNumber(cacheWrite)}` : null,
  ]
    .filter(Boolean)
    .join(", ")
  return (
    <span title={split}>
      {input}
      <span className="sr-only">, {split}</span>
    </span>
  )
}
