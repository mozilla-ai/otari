import { Skeleton } from "@/design-system/feedback/Skeleton"
import { formatNumber } from "@/shared/helpers/format"
import { type Reading, unitFor } from "./chartMath"

function Swatch({ fill }: { fill: string }) {
  return (
    <svg aria-hidden="true" viewBox="0 0 8 8" className="h-2 w-2 shrink-0">
      <rect width="8" height="8" fill={fill} />
    </svg>
  )
}

function Chip({
  fill,
  value,
  word,
}: {
  fill: string
  value: number
  word: string
}) {
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
      <Swatch fill={fill} />
      <span className="text-mono-caption tabular-nums text-foreground">
        {formatNumber(value)}
      </span>
      <span className="text-caption">{word}</span>
    </span>
  )
}

/**
 * The totals chips: what the window holds, or what the
 * bucket or range under the pointer holds. `reading === null` is still loading.
 */
export function ChartReadout({
  reading,
  label,
  unit,
}: {
  reading: Reading | null
  label?: string
  unit: string
}) {
  if (reading === null) {
    return (
      <span className="inline-flex items-center gap-4">
        <Skeleton className="h-3 w-24" />
        <Skeleton className="h-3 w-16" />
      </span>
    )
  }
  const word = reading.peak ? "at peak" : unitFor(reading.count, unit)
  const failed = reading.failed
  return (
    <span className="inline-flex flex-wrap items-center gap-x-4 gap-y-0.5">
      {label ? (
        <span className="whitespace-nowrap text-caption text-subtle">
          {label}
        </span>
      ) : null}
      <Chip fill="var(--color-primary)" value={reading.count} word={word} />
      {failed === null ? null : (
        <Chip
          fill={failed > 0 ? "var(--color-danger)" : "var(--color-text-subtle)"}
          value={failed}
          word="failed"
        />
      )}
    </span>
  )
}
