import { afterEach, describe, expect, it } from "vitest"
import {
  isPeriodKey,
  sessionsSearchFor,
  windowOf,
} from "@/features/spend/period"

// Noon on the ninth, local time, so the day arithmetic does not sit on a boundary.
const NOW = new Date(2026, 10, 9, 12, 0, 0).getTime()

const ORIGINAL_TZ = process.env.TZ
afterEach(() => {
  if (ORIGINAL_TZ === undefined) delete process.env.TZ
  else process.env.TZ = ORIGINAL_TZ
})

describe("windowOf", () => {
  it("starts this month on the first, local time", () => {
    const window = windowOf("month", NOW)
    expect(new Date(window.start).getTime()).toBe(
      new Date(2026, 10, 1).getTime(),
    )
    expect(window.days).toBe(9)
    expect(window.label).toBe("Nov 1 to Nov 9")
  })

  it("starts a rolling window on a whole day, today included", () => {
    const window = windowOf("7d", NOW)
    expect(new Date(window.start).getTime()).toBe(
      new Date(2026, 10, 3).getTime(),
    )
    expect(window.days).toBe(7)
  })

  it("counts calendar days across a daylight-saving change", () => {
    // Rome moves its clocks forward on March 29, 2026, so the seven local days
    // up to half past midnight on the 31st are an hour short of six days and
    // a half hour: still seven days of the calendar.
    process.env.TZ = "Europe/Rome"
    const now = new Date(2026, 2, 31, 0, 30).getTime()
    expect(windowOf("7d", now).days).toBe(7)
  })

  it("compares against the same length just before", () => {
    const window = windowOf("30d", NOW)
    const start = new Date(window.start).getTime()
    const previous = new Date(window.previousStart).getTime()
    expect(start - previous).toBe(NOW - start)
  })
})

describe("sessionsSearchFor", () => {
  it("opens Sessions from the window's own start, so it counts the same days", () => {
    // Sessions reads `range=7d` as now minus 168 hours, which starts mid-day
    // six days back where the figure starts at that day's local midnight.
    const week = windowOf("7d", NOW)
    expect(sessionsSearchFor(week)).toEqual({ start_date: week.start })
    const month = windowOf("month", NOW)
    expect(sessionsSearchFor(month)).toEqual({
      start_date: month.start,
    })
    const quarter = windowOf("90d", NOW)
    expect(sessionsSearchFor(quarter)).toEqual({
      start_date: quarter.start,
    })
  })
})

describe("isPeriodKey", () => {
  it("accepts the four windows and nothing else", () => {
    expect(isPeriodKey("month")).toBe(true)
    expect(isPeriodKey("90d")).toBe(true)
    expect(isPeriodKey("24h")).toBe(false)
    expect(isPeriodKey("")).toBe(false)
  })
})
