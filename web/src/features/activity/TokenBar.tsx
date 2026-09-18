import type { UsageEntry } from "@/client"
import { formatNumber } from "@/shared/helpers/format"
import {
  buildTokenComposition,
  formatTokenCount,
  TOKEN_SEGMENTS,
} from "./activityModel"

// The total plus a thin stacked bar of its composition. Widths are SVG rect
// attributes in a 100-unit viewBox (a dynamic Tailwind `w-[n%]` would not survive
// the JIT scanner, and inline styles are out). Proportions are exact, so a segment
// under a percent or so lands sub-pixel; the tooltip carries the real numbers.
//
// The number shown is the sum of the segments, so the cell is internally
// consistent and reads as "tokens billed". For an additive-convention row that is
// higher than the raw `total_tokens` column (which excludes the cache buckets);
// the raw fields stay visible, unchanged, in the detail panel.
export function TokenBar({ entry }: { entry: UsageEntry }) {
  const composition = buildTokenComposition(entry)
  if (composition === null) {
    return (
      <span className="tabular-nums">
        {formatTokenCount(entry.total_tokens)}
      </span>
    )
  }
  const parts = TOKEN_SEGMENTS.map((segment) => ({
    ...segment,
    value: composition[segment.key],
  }))
  const summary = parts
    .filter((part) => part.value > 0)
    .map((part) => `${part.label} ${formatNumber(part.value)}`)
    .join(", ")

  let offset = 0
  const rects = parts.map((part) => {
    const width = (part.value / composition.total) * 100
    const rect = { ...part, x: offset, width }
    offset += width
    return rect
  })

  return (
    <span className="inline-flex flex-col items-end gap-1.5" title={summary}>
      <span className="tabular-nums">{formatNumber(composition.total)}</span>
      <svg
        viewBox="0 0 100 4"
        preserveAspectRatio="none"
        role="img"
        aria-label={`Token composition: ${summary}`}
        className="h-1.5 w-20 overflow-hidden bg-primary-subtle"
      >
        {rects
          .filter((rect) => rect.width > 0)
          .map((rect) => (
            <rect
              key={rect.key}
              x={rect.x}
              y={0}
              width={rect.width}
              height={4}
              fill={rect.fill}
            />
          ))}
      </svg>
    </span>
  )
}
