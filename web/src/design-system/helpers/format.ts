/**
 * The two display formatters the components here need, and nothing else.
 *
 * The application's formatter module (`@/shared/helpers/format`) is still the
 * one a page reaches for, and still the only one DESIGN.md documents. These two
 * live here because `TrendChip` and `RefreshButton` need them and this layer
 * cannot import that one; the rest of that module (`formatUsd`, `formatTokens`,
 * `formatCost`, `formatContext`) stays there, because a spend figure and a
 * token count are this product's vocabulary rather than a design system's.
 *
 * There is one implementation of each: `@/shared/helpers/format` re-exports
 * these under the names it already published, so no call site moved and there
 * is no second copy to drift.
 */

/** A fraction as a percentage, to one decimal place. */
export function formatPct(fraction: number): string {
  return `${(fraction * 100).toFixed(1)}%`
}

/**
 * A relative time, compact: "3m ago", "2h ago", "5d ago", "2mo ago", "1y ago".
 *
 * Compact is the product's voice for this everywhere, which is a copy decision
 * with a layout consequence: "6 minutes ago" needed 130px of column where "6m
 * ago" fits 120, and a table lane widened to fit a phrase is a layout problem
 * wearing a copy costume. One implementation rather than two, because Activity
 * had grown its own and the two shapes were visibly different on pages sitting
 * one click apart.
 *
 * Deliberately not `Intl.RelativeTimeFormat`: its narrow style still prints
 * "6 min. ago" and its unit thresholds are not ours to choose.
 */
export function formatRelative(
  iso: string | null | undefined,
  now: number = Date.now(),
): string {
  if (!iso) {
    return "never"
  }
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) {
    return iso
  }
  const seconds = Math.round((now - date.getTime()) / 1000)
  // A clock skewed a few seconds ahead of the server is common and "in 2s" is
  // never what an operator wants to read about a request that already landed,
  // so the future collapses to the present rather than being spelled out.
  if (seconds < 0) return "just now"
  if (seconds < 60) return `${seconds}s ago`
  const minutes = Math.round(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.round(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.round(hours / 24)
  if (days < 30) return `${days}d ago`
  // The coarse buckets floor where the finer ones round, so "1y ago" covers the
  // whole year it names rather than a value at 18 months reading as two.
  const months = Math.floor(days / 30)
  if (months < 12) return `${months}mo ago`
  return `${Math.floor(months / 12)}y ago`
}
