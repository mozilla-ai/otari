import type { UsageEntry } from "@/client"
import { formatNumber } from "@/shared/helpers/format"
import {
  buildTokenComposition,
  formatTokenLane,
  isCompactCount,
} from "./activityModel"
import { LaneValue } from "./LaneValue"

// Billed input, cache reads and writes included, so Input and Output add up to
// the row's billed tokens. On a cached agent call the cache is most of that
// number, so the split is the title and is read out after it; the detail panel
// has every raw field.
export function InputTokens({ entry }: { entry: UsageEntry }) {
  const composition = buildTokenComposition(entry)
  const total = composition
    ? composition.fresh + composition.cacheRead + composition.cacheWrite
    : entry.prompt_tokens
  const display = formatTokenLane(total)
  const exact = total === null ? display : formatNumber(total)
  if (
    composition === null ||
    (!composition.cacheRead && !composition.cacheWrite)
  ) {
    return total !== null && isCompactCount(total) ? (
      <LaneValue display={display} exact={exact} />
    ) : (
      display
    )
  }
  const { fresh, cacheRead, cacheWrite } = composition
  const split = [
    `Fresh input ${formatNumber(fresh)}`,
    cacheRead ? `cache read ${formatNumber(cacheRead)}` : null,
    cacheWrite ? `cache write ${formatNumber(cacheWrite)}` : null,
  ]
    .filter(Boolean)
    .join(", ")
  return <LaneValue display={display} exact={exact} detail={split} />
}
