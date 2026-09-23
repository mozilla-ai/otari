// These three are re-exported rather than defined here: they carry no product
// vocabulary, so components in the design system need them, and that layer may
// not import this one. This module stays the single formatter module a page
// reaches for (DESIGN.md, "Where things come from"), so the names it published
// are unchanged and there is one implementation of each.
export {
  formatNumber,
  formatPct,
  formatRelative,
} from "@/design-system/helpers/format"

const usdPerRequest = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  minimumFractionDigits: 2,
  maximumFractionDigits: 4,
})

// What one request or one charge line cost. Four decimals throughout rather
// than only below a cent: a gateway's per-request costs sit in the hundredths
// and thousandths, so 2.34 cents rendered as "$0.02" drops the two digits an
// operator reading a request log is there for. Two decimals at least, so a
// whole-dollar cost still reads as money.
export function formatCost(value: number | null | undefined): string {
  if (value == null) {
    return "$0.00"
  }
  return usdPerRequest.format(value)
}

// A per-million rate, as opposed to a spend. Same precision as a cost and a
// separate name on purpose: $0.075 per million is a real published rate and
// "$0.08" is a figure nobody set, so the reason these carry four decimals is
// not the reason a cost does, and one moving should not drag the other.
export function formatRate(value: number): string {
  return usdPerRequest.format(value)
}

// Below what four decimals can show, fall back to significant digits. A
// per-call rate is routinely smaller than a per-million one: $0.00002 per
// search renders as "$0.0000" above and reads as free.
const usdSignificant = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumSignificantDigits: 3,
})

export function formatUnitRate(value: number): string {
  if (value === 0) return usdPerRequest.format(0)
  return value < 0.0001
    ? usdSignificant.format(value)
    : usdPerRequest.format(value)
}

// A millisecond duration an operator reads at a glance: "820 ms", "1.50 s".
// `undefined` rather than a placeholder when the row recorded none, so a table
// cell can render the em dash that keeps it aligned and a stat card can drop
// the figure instead.
export function formatLatency(
  ms: number | null | undefined,
): string | undefined {
  if (ms == null) {
    return undefined
  }
  if (ms < 1000) {
    return `${Math.round(ms)} ms`
  }
  return `${(ms / 1000).toFixed(2)} s`
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

// The heading a dated row sits under in a history list: "Today", "Yesterday",
// or the date, with the year only when it is not the current one.
//
// `now` is a parameter rather than a `new Date()` read inside, so a list left
// open across midnight relabels when its caller re-reads the clock instead of
// keeping yesterday's rows under "Today" until something else rerenders it.
export function formatDateGroup(
  iso: string | null | undefined,
  now: Date = new Date(),
): string {
  if (!iso) {
    return "\u2014"
  }
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) {
    return iso
  }
  const yesterday = new Date(now)
  yesterday.setDate(now.getDate() - 1)
  if (date.toDateString() === now.toDateString()) return "Today"
  if (date.toDateString() === yesterday.toDateString()) return "Yesterday"
  return date.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    ...(date.getFullYear() !== now.getFullYear()
      ? { year: "numeric" as const }
      : {}),
  })
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

const scoreFormat = new Intl.NumberFormat("en-US", { maximumFractionDigits: 3 })

// A guardrail vendor's score, which is whatever scale that vendor uses: shown
// as given, to three places, and never as a percentage it may not be.
export function formatScore(value: number): string {
  return scoreFormat.format(value)
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
