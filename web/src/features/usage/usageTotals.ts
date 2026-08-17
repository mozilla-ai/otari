import type { UsageSeriesPoint, UsageTotals } from "@/client";

// Derivations shared by the Usage page and the share card. They lived on the page
// while it was their only consumer; the card publishes the same numbers to the
// outside world, so a divergence here would mean the page and the posted image
// disagree about the same window.

// Billed token view: input (incl. both cache buckets) + output, falling back to
// the raw provider total when the composition fields are absent (an older
// gateway behind `vite dev`). Null when there are no totals at all.
export function billedTokenTotal(totals: UsageTotals | undefined): number | undefined {
  if (totals === undefined) {
    return undefined;
  }
  const billedInput = totals.billed_input_tokens;
  if (billedInput === undefined) {
    return totals.total_tokens;
  }
  return billedInput + (totals.billed_output_tokens ?? totals.completion_tokens);
}

// Cache sums from the series composition rather than the raw totals columns: the
// raw sums follow each provider's reporting convention, while the series is
// meter-normalized. One source keeps the headline, its trendline, and the hint in
// agreement.
export function cacheSums(points: UsageSeriesPoint[]): { input: number; read: number; write: number } {
  let input = 0;
  let read = 0;
  let write = 0;
  for (const p of points) {
    input += p.input_tokens ?? 0;
    read += p.cache_read_tokens ?? 0;
    write += p.cache_write_tokens ?? 0;
  }
  return { input, read, write };
}

export function cacheHitRate(points: UsageSeriesPoint[]): number | undefined {
  const { input, read } = cacheSums(points);
  return input > 0 ? read / input : undefined;
}

// Latency is nullable on the wire (null when no row recorded one). The page
// renders the em-dash placeholder to keep table cells aligned; the card drops the
// stat instead, so this returns undefined rather than a placeholder and each
// surface decides how to show "no value".
export function formatLatency(ms: number | null): string | undefined {
  if (ms === null) {
    return undefined;
  }
  if (ms < 1000) {
    return `${Math.round(ms)} ms`;
  }
  return `${(ms / 1000).toFixed(2)} s`;
}

// Whether a published cost figure needs its "N unpriced" caveat. `undefined`
// means the gateway predates the field, which is *more* likely to be missing
// prices, not less, so unknown counts as needing the caveat. A truthy check on
// the raw field would silently drop it exactly where it matters most.
export function costNeedsCaveat(totals: UsageTotals | undefined): boolean {
  if (totals === undefined) {
    return false;
  }
  return totals.unpriced_requests === undefined || totals.unpriced_requests > 0;
}
