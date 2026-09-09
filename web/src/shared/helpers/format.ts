// `formatPct` and `formatRelative` are re-exported rather than defined here:
// they carry no product vocabulary, so `TrendChip` and `RefreshButton` in the
// design system need them, and that layer may not import this one. This module
// stays the single formatter module a page reaches for (DESIGN.md, "Where
// things come from"), so the names it published are unchanged and there is one
// implementation of each.
export { formatPct, formatRelative } from "@/design-system/helpers/format"

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

// Period-over-period change. null when there is no comparable previous value
// (unbounded range, or a previous value of zero which would divide by zero).
export function deltaFraction(
  current: number,
  previous: number | undefined,
): number | null {
  if (previous === undefined || previous === 0) return null
  return (current - previous) / previous
}
