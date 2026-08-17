import type { UsageBucket } from "@/client"

// Shared time-range vocabulary for the Usage and Activity pages. Both pages used
// to keep private copies of the preset list (which drifted: "All" meant 30 days
// on Usage but all-time on Activity). Centralizing the presets and the window
// math here keeps the two pages honest, and feeds the shared activity-timeline
// selector (a request-volume histogram you drag across to pick a range).
//
// Windows are absolute instants and are read in UTC, matching the summary
// endpoint's UTC bucketing and the existing "times in UTC" trend caption. The
// timeline's buckets are UTC-aligned, so a picked range lines up with the bars.

export const HOUR_S = 3_600
export const DAY_S = 86_400

// The widest window we actually *request* (the 12-month preset and the unbounded
// "All" histogram). The server caps a single summary window at 366 days
// (`_MAX_SUMMARY_SPAN`), and we stop a day short of it on purpose: the client
// anchors `start` while the server resolves `end` at request time, so a full-366
// request always exceeds the cap by the round-trip delta (and shrinks further
// behind a slow client clock) and is silently pulled forward. 365 days leaves
// headroom and never trips the clamp, so the preset spans what it claims to.
export const YEAR_SPAN_S = 365 * DAY_S

export interface RangePreset {
  key: string
  label: string
  // A rolling lookback of this many seconds ("now minus N"). `null` is an
  // explicit unbounded ("All") window; only Activity uses it, because its raw
  // list endpoint treats an omitted start as all-time. Usage's summary endpoint
  // reads an omitted start as a 30-day *default*, so its preset list carries no
  // null: an unbounded Usage window is neither representable nor meaningful.
  seconds: number | null
  bucket: UsageBucket
}

// Sub-day windows bucket hourly; longer ranges bucket daily. Usage adds bounded
// long-range presets (90d, a year) in place of the old rolling "All", so the
// page still reaches the full span the server will serve without an unbounded,
// clock-rolling window whose total silently changes day to day.
export const USAGE_PRESETS: RangePreset[] = [
  { key: "1h", label: "Last hour", seconds: HOUR_S, bucket: "hour" },
  { key: "24h", label: "24h", seconds: DAY_S, bucket: "hour" },
  { key: "7d", label: "7d", seconds: 7 * DAY_S, bucket: "day" },
  { key: "30d", label: "30d", seconds: 30 * DAY_S, bucket: "day" },
  { key: "90d", label: "90d", seconds: 90 * DAY_S, bucket: "day" },
  { key: "12mo", label: "12mo", seconds: YEAR_SPAN_S, bucket: "day" },
]

// A spend investigation is usually monthly.
export const USAGE_DEFAULT_KEY = "30d"

// Activity keeps a truthful "All": the raw list endpoint applies no default and
// no clamp, so an omitted start really is all-time there.
export const ACTIVITY_PRESETS: RangePreset[] = [
  { key: "1h", label: "1h", seconds: HOUR_S, bucket: "hour" },
  { key: "24h", label: "24h", seconds: DAY_S, bucket: "hour" },
  { key: "7d", label: "7d", seconds: 7 * DAY_S, bucket: "day" },
  { key: "30d", label: "30d", seconds: 30 * DAY_S, bucket: "day" },
  { key: "all", label: "All", seconds: null, bucket: "day" },
]

export const ACTIVITY_DEFAULT_KEY = "24h"

// The sentinel key the preset row highlights while an explicit range is active
// (drag-selected or zoomed). Not a preset itself, so it never appears in a list.
export const CUSTOM_KEY = "custom"

export function findPreset(
  presets: RangePreset[],
  key: string,
): RangePreset | undefined {
  return presets.find((p) => p.key === key)
}

export function isoAgo(seconds: number, now: number = Date.now()): string {
  return new Date(now - seconds * 1000).toISOString()
}

export function bucketDurationMs(bucket: UsageBucket): number {
  return (bucket === "hour" ? HOUR_S : DAY_S) * 1000
}

// Bucket granularity for a window, chosen from its length rather than from
// whether the range is "custom": a three-hour window should still bucket hourly
// instead of collapsing to a single point. An open-ended window (no explicit
// end) is measured against `now`.
export function bucketForWindow(
  startIso: string,
  endIso: string | undefined,
  now: number = Date.now(),
): UsageBucket {
  const start = new Date(startIso).getTime()
  const end = endIso ? new Date(endIso).getTime() : now
  return end - start <= DAY_S * 1000 ? "hour" : "day"
}

// The absolute window covered by buckets [i..j] of a timeline series (order of
// i/j does not matter). Start is the first bucket's start; end is the exclusive
// upper bound (last bucket's start plus one bucket), so the whole last bucket is
// included. Indices are clamped to the series bounds.
export function rangeFromBuckets(
  starts: string[],
  i: number,
  j: number,
  bucket: UsageBucket,
): { startIso: string; endIso: string } | null {
  if (starts.length === 0) return null
  const lo = Math.max(0, Math.min(i, j))
  const hi = Math.min(starts.length - 1, Math.max(i, j))
  const startMs = new Date(starts[lo]).getTime()
  const endMs = new Date(starts[hi]).getTime() + bucketDurationMs(bucket)
  return {
    startIso: new Date(startMs).toISOString(),
    endIso: new Date(endMs).toISOString(),
  }
}

// Map an active window onto bucket indices of a timeline (extent) series, for
// positioning the brush slider. `startIndex` is the last bucket starting at or
// before the window start; `endIndex` is the last bucket starting before the
// (exclusive) window end. An absent bound extends to that end of the series.
export function bucketIndexRange(
  starts: string[],
  startIso: string | undefined,
  endIso: string | undefined,
): { startIndex: number; endIndex: number } {
  const n = starts.length
  if (n === 0) return { startIndex: 0, endIndex: 0 }
  const ms = starts.map((s) => new Date(s).getTime())
  let startIndex = 0
  if (startIso) {
    const t = new Date(startIso).getTime()
    for (let i = 0; i < n; i++) {
      if (ms[i] <= t) startIndex = i
    }
  }
  let endIndex = n - 1
  if (endIso) {
    const t = new Date(endIso).getTime()
    endIndex = 0
    for (let i = 0; i < n; i++) {
      if (ms[i] < t) endIndex = i
    }
  }
  if (endIndex < startIndex) endIndex = startIndex
  return { startIndex, endIndex }
}

// ---------- effective-window caption ----------

function formatDay(ms: number, withYear: boolean): string {
  return new Date(ms).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    ...(withYear ? { year: "numeric" } : {}),
    timeZone: "UTC",
  })
}

// A human caption for the window the server actually aggregated over, so a
// default or a clamp is never invisible. Sourced from the echoed start/end
// instants and rendered in UTC calendar terms (end shown inclusively).
export function formatWindowLabel(
  startIso: string | undefined,
  endIso: string | undefined,
): string {
  if (!startIso && !endIso) {
    return "All time"
  }
  if (startIso && !endIso) {
    return `Since ${formatDay(new Date(startIso).getTime(), true)}`
  }
  if (!startIso && endIso) {
    return `Up to ${formatDay(new Date(endIso).getTime() - 1, true)}`
  }
  const startMs = new Date(startIso as string).getTime()
  // Inclusive last day: our windows end at the next bucket's start (exclusive).
  const endMs = new Date(endIso as string).getTime() - 1
  const sameDay = formatDay(startMs, true) === formatDay(endMs, true)
  if (sameDay) {
    return formatDay(startMs, true)
  }
  const sameYear =
    new Date(startMs).getUTCFullYear() === new Date(endMs).getUTCFullYear()
  return `${formatDay(startMs, !sameYear)} – ${formatDay(endMs, true)}`
}
