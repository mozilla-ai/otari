import { afterEach, describe, expect, it } from "vitest"
import {
  comparedWith,
  DEFAULT_PERIOD,
  figureLabel,
  isPeriodKey,
  PERIODS,
  periodFromUrl,
  sessionsSearchFor,
  usageRange,
  type WindowKind,
  windowOf,
  windowSpan,
  windowWords,
} from "@/features/spend/period"

// Noon on the ninth, local time, so the day arithmetic does not sit on a boundary.
const NOW = new Date(2026, 10, 9, 12, 0, 0).getTime()
const HOUR_MS = 3_600_000

const local = (...parts: [number, number, number, number?, number?]) =>
  new Date(...(parts as [number, number, number, number, number])).getTime()
const ms = (iso: string) => new Date(iso).getTime()

const ORIGINAL_TZ = process.env.TZ
afterEach(() => {
  if (ORIGINAL_TZ === undefined) delete process.env.TZ
  else process.env.TZ = ORIGINAL_TZ
})

describe("the picker", () => {
  it("offers the six windows in Mixpanel's order, with their short labels", () => {
    expect(PERIODS.map((period) => period.label)).toEqual([
      "Today",
      "Yesterday",
      "7D",
      "30D",
      "3M",
      "6M",
    ])
  })

  it("opens on 30 days", () => {
    expect(DEFAULT_PERIOD).toBe("30d")
  })
})

describe("periodFromUrl", () => {
  it("reads each window it offers", () => {
    for (const { value } of PERIODS) expect(periodFromUrl(value)).toBe(value)
  })

  it("reads an old 90-day link as three months", () => {
    expect(periodFromUrl("90d")).toBe("3m")
  })

  it("falls back to the default for anything it does not offer", () => {
    for (const value of [
      "month",
      "custom",
      "",
      "24h",
      "7D",
      "garbage",
      "__proto__",
      "toString",
    ]) {
      expect(periodFromUrl(value)).toBe(DEFAULT_PERIOD)
    }
  })

  it("accepts only the picker's keys as period keys", () => {
    expect(isPeriodKey("today")).toBe(true)
    expect(isPeriodKey("6m")).toBe(true)
    expect(isPeriodKey("month")).toBe(false)
    expect(isPeriodKey("90d")).toBe(false)
    expect(isPeriodKey("")).toBe(false)
  })
})

describe("windowOf", () => {
  it("reads today from midnight to now, by the hour", () => {
    const window = windowOf("today", NOW)
    expect(ms(window.start)).toBe(local(2026, 10, 9))
    expect(window.end).toBe("")
    expect(window.days).toBe(1)
    expect(window.label).toBe("Nov 9")
    expect(window.bucket).toBe("hour")
  })

  it("compares today with yesterday up to the same time of day", () => {
    // Seconds in: the comparison ends on the minute, so its key holds still.
    const window = windowOf("today", NOW + 37_000)
    expect(ms(window.previousStart)).toBe(local(2026, 10, 8))
    expect(ms(window.previousEnd)).toBe(local(2026, 10, 8, 12, 0))
  })

  it("reads yesterday whole, midnight to midnight, by the hour", () => {
    const window = windowOf("yesterday", NOW)
    expect(ms(window.start)).toBe(local(2026, 10, 8))
    expect(ms(window.end)).toBe(local(2026, 10, 9))
    expect(window.days).toBe(1)
    expect(window.label).toBe("Nov 8")
    expect(window.bucket).toBe("hour")
    // The day before it, whole.
    expect(ms(window.previousStart)).toBe(local(2026, 10, 7))
    expect(ms(window.previousEnd)).toBe(local(2026, 10, 8))
  })

  it("starts a rolling window on a whole day, today included", () => {
    const week = windowOf("7d", NOW)
    expect(ms(week.start)).toBe(local(2026, 10, 3))
    expect(week.end).toBe("")
    expect(week.days).toBe(7)
    expect(week.label).toBe("Nov 3 to Nov 9")
    expect(week.bucket).toBe("day")
    const month = windowOf("30d", NOW)
    expect(ms(month.start)).toBe(local(2026, 9, 11))
    expect(month.days).toBe(30)
  })

  it("starts 3 and 6 months back on the same date", () => {
    const quarter = windowOf("3m", NOW)
    expect(ms(quarter.start)).toBe(local(2026, 7, 9))
    expect(quarter.end).toBe("")
    expect(quarter.days).toBe(93)
    expect(quarter.bucket).toBe("day")
    const half = windowOf("6m", NOW)
    expect(ms(half.start)).toBe(local(2026, 4, 9))
    expect(half.days).toBe(185)
  })

  it("clamps a month back to the last day of a shorter month", () => {
    const may31 = local(2026, 4, 31, 12)
    expect(ms(windowOf("3m", may31).start)).toBe(local(2026, 1, 28))
    // A leap year has the 29th.
    expect(ms(windowOf("3m", local(2028, 4, 31, 12)).start)).toBe(
      local(2028, 1, 29),
    )
    expect(ms(windowOf("6m", local(2026, 7, 31, 12)).start)).toBe(
      local(2026, 1, 28),
    )
    // Across the turn of the year.
    expect(ms(windowOf("3m", local(2027, 2, 31, 12)).start)).toBe(
      local(2026, 11, 31),
    )
    expect(ms(windowOf("6m", local(2027, 0, 15, 12)).start)).toBe(
      local(2026, 6, 15),
    )
  })

  it("compares an open window with the same length just before it", () => {
    for (const kind of ["7d", "30d", "3m", "6m", "month"] as WindowKind[]) {
      const window = windowOf(kind, NOW)
      const start = ms(window.start)
      expect(ms(window.previousEnd)).toBe(start)
      expect(start - ms(window.previousStart)).toBe(NOW - start)
    }
  })

  it("starts this month on the first, local time", () => {
    const window = windowOf("month", NOW)
    expect(ms(window.start)).toBe(local(2026, 10, 1))
    expect(window.end).toBe("")
    expect(window.days).toBe(9)
    expect(window.label).toBe("Nov 1 to Nov 9")
    expect(window.bucket).toBe("day")
  })
})

describe("windowOf across a daylight-saving change", () => {
  // Rome moves its clocks forward on March 29, 2026 and back on October 25.
  // Each test first proves the zone took, or it would pass in UTC and prove
  // nothing.
  const offsetsDiffer = (a: number, b: number) =>
    new Date(a).getTimezoneOffset() !== new Date(b).getTimezoneOffset()

  it("reads a 23-hour yesterday as one day", () => {
    process.env.TZ = "Europe/Rome"
    const now = local(2026, 2, 30, 12)
    const window = windowOf("yesterday", now)
    expect(offsetsDiffer(ms(window.start), ms(window.end))).toBe(true)
    expect(ms(window.end) - ms(window.start)).toBe(23 * HOUR_MS)
    expect(window.days).toBe(1)
    expect(new Date(window.start).getHours()).toBe(0)
    expect(new Date(window.end).getHours()).toBe(0)
  })

  it("reads a 25-hour yesterday as one day", () => {
    process.env.TZ = "Europe/Rome"
    const window = windowOf("yesterday", local(2026, 9, 26, 12))
    expect(offsetsDiffer(ms(window.start), ms(window.end))).toBe(true)
    expect(ms(window.end) - ms(window.start)).toBe(25 * HOUR_MS)
    expect(window.days).toBe(1)
  })

  it("starts today at local midnight on the short day, against the same time yesterday", () => {
    process.env.TZ = "Europe/Rome"
    const now = local(2026, 2, 29, 12)
    const window = windowOf("today", now)
    expect(offsetsDiffer(ms(window.start), now)).toBe(true)
    expect(new Date(window.start).getHours()).toBe(0)
    expect(window.days).toBe(1)
    // Noon yesterday, by the clock: 23 hours earlier, not 24.
    expect(new Date(window.previousEnd).getHours()).toBe(12)
    expect(now - ms(window.previousEnd)).toBe(23 * HOUR_MS)
  })

  it("counts calendar days across the change", () => {
    // Seven local days up to half past midnight on the 31st are an hour short
    // of six days and a half: still seven days of the calendar.
    process.env.TZ = "Europe/Rome"
    const now = local(2026, 2, 31, 0, 30)
    const window = windowOf("7d", now)
    expect(offsetsDiffer(ms(window.start), now)).toBe(true)
    expect(window.days).toBe(7)
  })
})

describe("the window's words", () => {
  const words = (kind: WindowKind) => {
    const window = windowOf(kind, NOW)
    return {
      sentence: windowWords(window),
      span: windowSpan(window),
      spent: figureLabel("Spent", window),
      versus: comparedWith(window),
    }
  }

  it("reads right in each place it lands", () => {
    expect(words("today")).toEqual({
      sentence: "today",
      span: "today",
      spent: "Spent today",
      versus: "vs yesterday",
    })
    expect(words("yesterday")).toEqual({
      sentence: "yesterday",
      span: "yesterday",
      spent: "Spent yesterday",
      // The ninth is a Monday, so yesterday is Sunday and the day before it Saturday.
      versus: "vs Saturday",
    })
    expect(words("7d")).toEqual({
      sentence: "in the last 7 days",
      span: "7 days",
      spent: "Spent, 7 days",
      versus: "vs prior 7 days",
    })
    expect(words("30d").sentence).toBe("in the last 30 days")
    expect(words("3m")).toEqual({
      sentence: "in the last 3 months",
      span: "3 months",
      spent: "Spent, 3 months",
      versus: "vs prior 3M",
    })
    expect(words("6m").spent).toBe("Spent, 6 months")
    expect(words("6m").versus).toBe("vs prior 6M")
    expect(words("month")).toEqual({
      sentence: "this month",
      span: "this month",
      spent: "Spent this month",
      versus: "vs prior 9 days",
    })
  })

  it("says one day, not 1 days, on the first of the month", () => {
    const window = windowOf("month", local(2026, 10, 1, 9))
    expect(comparedWith(window)).toBe("vs prior 1 day")
  })
})

describe("usageRange and sessionsSearchFor", () => {
  it("names no end for a window that runs to now", () => {
    for (const kind of ["today", "7d", "30d", "3m", "6m", "month"] as const) {
      const window = windowOf(kind, NOW)
      expect(usageRange(window)).toEqual({ start_date: window.start })
      expect(sessionsSearchFor(window)).toEqual({ start_date: window.start })
    }
  })

  it("carries yesterday's end, so it never counts today", () => {
    const window = windowOf("yesterday", NOW)
    expect(usageRange(window)).toEqual({
      start_date: window.start,
      end_date: window.end,
    })
    // Sessions reads the same two keys.
    expect(sessionsSearchFor(window)).toEqual({
      start_date: window.start,
      end_date: window.end,
    })
  })
})
