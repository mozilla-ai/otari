/**
 * The windows money is read over. "This month" is the calendar
 * month so far, which is what an invoice and a monthly limit both count; the
 * others are rolling, for looking back across a month boundary.
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

export type PeriodKey = "month" | "7d" | "30d" | "90d"

export const PERIODS: { value: PeriodKey; label: string }[] = [
  { value: "month", label: "This month" },
  { value: "7d", label: "7 days" },
  { value: "30d", label: "30 days" },
  { value: "90d", label: "90 days" },
]

const ROLLING_DAYS: Record<Exclude<PeriodKey, "month">, number> = {
  "7d": 7,
  "30d": 30,
  "90d": 90,
}

export function isPeriodKey(value: string): value is PeriodKey {
  return PERIODS.some((period) => period.value === value)
}

export interface SpendWindow {
  /** ISO start of the window; it runs to now. */
  start: string
  /** The window of the same length just before it, for "vs previous". */
  previousStart: string
  /** Days the window has run, never less than one, for a per-day figure. */
  days: number
  /** "Nov 1 to Nov 9", for a figure's subline. */
  label: string
}

const DAY_LABEL = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
})

/**
 * `now` is passed in so a page reads one clock for its whole life: a window
 * built from the render's own `Date.now()` is a new query on every render.
 */
export function windowOf(period: PeriodKey, now: number): SpendWindow {
  const today = new Date(now)
  const start =
    period === "month"
      ? new Date(today.getFullYear(), today.getMonth(), 1).getTime()
      : // Whole local days, so the first bar of a chart is a full day.
        new Date(
          today.getFullYear(),
          today.getMonth(),
          today.getDate() - (ROLLING_DAYS[period] - 1),
        ).getTime()
  // Calendar days, today included: a day across a daylight-saving change is
  // 23 or 25 hours long, so dividing elapsed time by 24 hours miscounts.
  const startOfToday = new Date(
    today.getFullYear(),
    today.getMonth(),
    today.getDate(),
  ).getTime()
  return {
    start: new Date(start).toISOString(),
    previousStart: new Date(start - (toMinute(now) - start)).toISOString(),
    days: Math.max(1, Math.round((startOfToday - start) / DAY_MS) + 1),
    label: `${DAY_LABEL.format(start)} to ${DAY_LABEL.format(now)}`,
  }
}

/** "this month", "in the last 7 days": the window inside a sentence. */
export const windowWords = (period: PeriodKey, window: SpendWindow) =>
  period === "month" ? "this month" : `in the last ${window.days} days`

/**
 * The filters that open Sessions on the same window: its own start, since a
 * Sessions preset (`range=7d`) runs back from now by the hour where a window
 * here starts at a local midnight.
 */
export function sessionsSearchFor(window: SpendWindow): Record<string, string> {
  return { start_date: window.start }
}
