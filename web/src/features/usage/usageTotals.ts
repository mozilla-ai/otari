import type { UsageSeriesPoint, UsageTotals } from "@/client"

// Billed token view: input (incl. both cache buckets) + output, falling back to
// the raw provider total when the composition fields are absent (an older
// gateway behind `vite dev`). Null when there are no totals at all.
export function billedTokenTotal(
  totals: UsageTotals | undefined,
): number | undefined {
  if (totals === undefined) {
    return undefined
  }
  const billedInput = totals.billed_input_tokens
  if (billedInput === undefined) {
    return totals.total_tokens
  }
  return billedInput + (totals.billed_output_tokens ?? totals.completion_tokens)
}

// Cache sums from the series composition rather than the raw totals columns: the
// raw sums follow each provider's reporting convention, while the series is
// meter-normalized. One source keeps the headline, its trendline, and the hint in
// agreement.
export function cacheSums(points: UsageSeriesPoint[]): {
  input: number
  read: number
  write: number
} {
  return points.reduce(
    (sums, p) => ({
      input: sums.input + (p.input_tokens ?? 0),
      read: sums.read + (p.cache_read_tokens ?? 0),
      write: sums.write + (p.cache_write_tokens ?? 0),
    }),
    { input: 0, read: 0, write: 0 },
  )
}
