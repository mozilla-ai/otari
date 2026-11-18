import type { UsageBucket } from "@/client"

/**
 * The windows money is read over, in the browser's local time. Home picks one
 * of `PERIODS`: today so far, yesterday whole, or a rolling stretch of whole
 * days that ends with today. "This month" is the calendar month so far, which
 * is what an invoice and a monthly limit both count, and what Keys and People
 * read.
 */

const DAY_MS = 86_400_000
const MINUTE_MS = 60_000

/**
 * `now` to the minute, for a window that runs back from it: a query keyed on
 * the millisecond is a new one on every mount, so returning to a page read it
 * again however fresh the last answer was.
 */
export function toMinute(now: number): number {
  return now - (now % MINUTE_MS)
}

export type PeriodKey = "today" | "yesterday" | "7d" | "30d" | "3m" | "6m"

/** Every window there is: the picker's, and the calendar month so far. */
export type WindowKind = PeriodKey | "month"

export const PERIODS: { value: PeriodKey; label: string }[] = [
  { value: "today", label: "Today" },
  { value: "yesterday", label: "Yesterday" },
  { value: "7d", label: "7D" },
  { value: "30d", label: "30D" },
  { value: "3m", label: "3M" },
  { value: "6m", label: "6M" },
]

/** Home's window when its URL names none, or names one it no longer offers. */
export const DEFAULT_PERIOD: PeriodKey = "30d"

const ROLLING_DAYS: Partial<Record<WindowKind, number>> = { "7d": 7, "30d": 30 }
const MONTHS_BACK: Partial<Record<WindowKind, number>> = { "3m": 3, "6m": 6 }

export function isPeriodKey(value: string): value is PeriodKey {
  return PERIODS.some((period) => period.value === value)
}

/** The window a URL's `period` names, or the default for one it does not. */
export function periodFromUrl(value: string): PeriodKey {
  // A link from before the rename: `90d` was the quarter.
  const key = value === "90d" ? "3m" : value
  return isPeriodKey(key) ? key : DEFAULT_PERIOD
}

export interface SpendWindow {
  kind: WindowKind
  /** ISO start of the window. */
  start: string
  /** ISO end, exclusive; "" for a window that runs to now. */
  end: string
  /** The window "vs previous" compares with. */
  previousStart: string
  previousEnd: string
  /** Calendar days the window spans, never less than one, for a per-day figure. */
  days: number
  /** "Nov 1 to Nov 9", or "Nov 9" for one day, for a figure's subline. */
  label: string
  /** The series' granularity: hours across a single day, days otherwise. */
  bucket: UsageBucket
}

const DAY_LABEL = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
})

const WEEKDAY = new Intl.DateTimeFormat("en-US", { weekday: "long" })

const iso = (ms: number) => new Date(ms).toISOString()

/**
 * `now` is passed in so a page reads one clock for its whole life: a window
 * built from the render's own `Date.now()` is a new query on every render.
 *
 * Every boundary is a local calendar date rather than a count of hours, so a
 * day across a daylight-saving change is the 23 or 25 hours it really is.
 */
export function windowOf(kind: WindowKind, now: number): SpendWindow {
  const clock = new Date(toMinute(now))
  const bounds = boundsOf(kind, clock)
  const today = dayOf(clock.getTime())
  // The last day the window covers: today, or the one before an end.
  const lastDay = bounds.end ? dayOf(bounds.end - 1) : today
  return {
    kind,
    start: iso(bounds.start),
    end: bounds.end ? iso(bounds.end) : "",
    previousStart: iso(bounds.previousStart),
    previousEnd: iso(bounds.previousEnd),
    // Rounded, because a day across a daylight-saving change is not 24 hours.
    days: Math.max(1, Math.round((lastDay - bounds.start) / DAY_MS) + 1),
    label:
      lastDay === bounds.start
        ? DAY_LABEL.format(bounds.start)
        : `${DAY_LABEL.format(bounds.start)} to ${DAY_LABEL.format(
            bounds.end ? lastDay : clock,
          )}`,
    bucket: kind === "today" || kind === "yesterday" ? "hour" : "day",
  }
}

/** Local midnight on the day `ms` falls on, `offset` days on. */
function dayOf(ms: number, offset = 0): number {
  const at = new Date(ms)
  return new Date(
    at.getFullYear(),
    at.getMonth(),
    at.getDate() + offset,
  ).getTime()
}

/** Where a window starts and ends (0 when it runs to now), and the one before it. */
function boundsOf(
  kind: WindowKind,
  clock: Date,
): { start: number; end: number; previousStart: number; previousEnd: number } {
  const now = clock.getTime()
  if (kind === "today") {
    return {
      start: dayOf(now),
      end: 0,
      // Yesterday up to this time of day, so a morning is not weighed against
      // a whole day.
      previousStart: dayOf(now, -1),
      previousEnd: new Date(
        clock.getFullYear(),
        clock.getMonth(),
        clock.getDate() - 1,
        clock.getHours(),
        clock.getMinutes(),
      ).getTime(),
    }
  }
  if (kind === "yesterday") {
    return {
      start: dayOf(now, -1),
      end: dayOf(now),
      previousStart: dayOf(now, -2),
      previousEnd: dayOf(now, -1),
    }
  }
  const rolling = ROLLING_DAYS[kind]
  const months = MONTHS_BACK[kind]
  const start =
    kind === "month"
      ? new Date(clock.getFullYear(), clock.getMonth(), 1).getTime()
      : months
        ? monthsBack(clock, months)
        : // Whole local days, today included, so a chart's first bar is a full day.
          dayOf(now, -((rolling ?? 1) - 1))
  // It runs to now, so the one before it is as long as it has run so far.
  return {
    start,
    end: 0,
    previousStart: start - (now - start),
    previousEnd: start,
  }
}

/**
 * Local midnight on the same date `months` back, or that month's last day
 * when it is shorter: three months back from May 31 is February 28 (29).
 */
function monthsBack(from: Date, months: number): number {
  const year = from.getFullYear()
  const month = from.getMonth() - months
  const lastOfMonth = new Date(year, month + 1, 0).getDate()
  return new Date(year, month, Math.min(from.getDate(), lastOfMonth)).getTime()
}

const plural = (count: number, one: string) =>
  `${count} ${one}${count === 1 ? "" : "s"}`

/** "today", "7 days", "3 months": how far the window reaches. */
export function windowSpan(window: SpendWindow): string {
  const { kind } = window
  if (kind === "today" || kind === "yesterday") return kind
  if (kind === "month") return "this month"
  const months = MONTHS_BACK[kind]
  return months ? plural(months, "month") : plural(window.days, "day")
}

/** "today", "in the last 7 days": the window inside a sentence. */
export function windowWords(window: SpendWindow): string {
  const { kind } = window
  return kind === "today" || kind === "yesterday" || kind === "month"
    ? windowSpan(window)
    : `in the last ${windowSpan(window)}`
}

/** A figure's label over the window: "Spent today", "Tokens, 7 days". */
export function figureLabel(noun: string, window: SpendWindow): string {
  const { kind } = window
  return kind === "today" || kind === "yesterday" || kind === "month"
    ? `${noun} ${windowSpan(window)}`
    : `${noun}, ${windowSpan(window)}`
}

/**
 * What "vs previous" weighs the window against, under the change: "vs
 * yesterday" (up to this time of day), "vs Sunday", "vs prior 7 days", "vs
 * prior 3M". Short, because the change holds its width in a figure a fifth of
 * the page wide.
 */
export function comparedWith(window: SpendWindow): string {
  if (window.kind === "today") return "vs yesterday"
  if (window.kind === "yesterday") {
    return `vs ${WEEKDAY.format(new Date(window.previousStart))}`
  }
  if (window.kind === "month") return `vs prior ${plural(window.days, "day")}`
  // In the picker's own words, which keeps the change inside its figure at 1280px.
  const months = MONTHS_BACK[window.kind]
  return months ? `vs prior ${months}M` : `vs prior ${windowSpan(window)}`
}

/**
 * The window as the usage API's filters. An open window names no end, so it
 * reads to now and its request is the one it always was.
 */
export function usageRange(window: SpendWindow): {
  start_date: string
  end_date?: string
} {
  return window.end
    ? { start_date: window.start, end_date: window.end }
    : { start_date: window.start }
}

/**
 * The filters that open Sessions on the same window: its own bounds, since a
 * Sessions preset (`range=7d`) runs back from now by the hour where a window
 * here starts at a local midnight.
 */
export function sessionsSearchFor(window: SpendWindow): Record<string, string> {
  return usageRange(window)
}
