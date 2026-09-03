export function formatNumber(value: number | null | undefined): string {
  if (value == null) {
    return "0"
  }
  return new Intl.NumberFormat("en-US").format(value)
}

export function formatCost(value: number | null | undefined): string {
  if (value == null) {
    return "$0.00"
  }
  // Show more precision for tiny per-request costs so they don't read as $0.00.
  const fractionDigits = value !== 0 && Math.abs(value) < 0.01 ? 4 : 2
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: fractionDigits,
  }).format(value)
}

// Compact token counts for context windows: 128000 -> "128K", 1000000 -> "1M".
// Returns an em-dash placeholder when unknown so table cells stay aligned.
export function formatContext(value: number | null | undefined): string {
  if (value == null) {
    return "—"
  }
  if (value >= 1_000_000) {
    const millions = value / 1_000_000
    return `${Number.isInteger(millions) ? millions : millions.toFixed(1)}M`
  }
  if (value >= 1000) {
    // Promote to "1M" rather than "1000K" when rounding lands on a thousand-K
    // (e.g. 999999 rounds to 1000K).
    const thousands = Math.round(value / 1000)
    return thousands >= 1000 ? "1M" : `${thousands}K`
  }
  return String(value)
}

const MONTH_ABBREVIATIONS = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "May",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Oct",
  "Nov",
  "Dec",
]

// models.dev release dates arrive as "YYYY-MM-DD" (occasionally just "YYYY-MM").
// Render a compact "Mon YYYY" for the table without pulling the value through a
// timezone-shifting Date parse. Returns an em-dash placeholder when unknown.
export function formatReleaseDate(value: string | null | undefined): string {
  if (!value) {
    return "—"
  }
  const match = /^(\d{4})-(\d{2})/.exec(value)
  if (!match) {
    return value
  }
  const monthIndex = Number(match[2]) - 1
  if (monthIndex < 0 || monthIndex > 11) {
    return match[1]
  }
  return `${MONTH_ABBREVIATIONS[monthIndex]} ${match[1]}`
}

// Date only, for table cells where the time of day carries nothing. Falls back
// to the raw string rather than rendering "Invalid Date", matching formatDateTime.
export function formatDate(iso: string | null | undefined): string {
  if (!iso) {
    return "—"
  }
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) {
    return iso
  }
  return date.toLocaleDateString()
}

export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) {
    return "—"
  }
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) {
    return iso
  }
  return date.toLocaleString()
}

// Compact USD for aggregate tiles: cents precision (not the per-request 4dp that
// formatCost uses), so four+ figure totals stay readable. Non-null: callers guard
// nullable per-request costs (e.g. `cost === null ? "—" : formatUsd(cost)`).
const usdCompact = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 2,
})

export function formatUsd(value: number): string {
  return usdCompact.format(value)
}

const usdWhole = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
})

// Dollars set as a headline rather than read off a table. From $100 up the cents
// are receipt detail: they add two of the widest glyphs on the line to carry a
// precision nobody checks at a glance, and on the share card that width comes
// straight out of the type size. Below $100 they still say something, since the
// difference between $4.10 and $4.99 is a quarter of the number.
export function formatUsdHeadline(value: number): string {
  return Math.abs(value) >= 100
    ? usdWhole.format(value)
    : usdCompact.format(value)
}

// Compact token counts for aggregate tiles: 12.4M / 84.2k / 512.
export function formatTokens(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}k`
  return String(value)
}

export function formatPct(fraction: number): string {
  return `${(fraction * 100).toFixed(1)}%`
}

// Period-over-period change. null when there is no comparable previous value
// (unbounded range, or a previous value of zero which would divide by zero).
export function deltaFraction(
  current: number,
  previous: number | undefined,
): number | null {
  if (previous === undefined || previous === 0) return null
  return (current - previous) / previous
}

/**
 * A relative time, compact: "3m ago", "2h ago", "5d ago".
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
  return `${Math.round(hours / 24)}d ago`
}
