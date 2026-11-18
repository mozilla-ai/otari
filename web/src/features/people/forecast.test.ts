import { describe, expect, it } from "vitest"
import { runsOutAt } from "@/features/people/forecast"

const HOUR = 3_600_000
const START = Date.parse("2026-11-01T00:00:00Z")
const END = Date.parse("2026-12-01T00:00:00Z")
const period = {
  periodStart: new Date(START).toISOString(),
  periodEnd: new Date(END).toISOString(),
}

describe("runsOutAt", () => {
  it("projects the period's pace so far onto what is left", () => {
    // $30 of $40 in ten days: $3 a day, so the last $10 lasts three more.
    const now = START + 240 * HOUR
    expect(runsOutAt({ spent: 30, max: 40, now, ...period })).toBe(
      now + 80 * HOUR,
    )
  })

  it("says nothing when the limit lasts the period", () => {
    const now = START + 240 * HOUR
    expect(runsOutAt({ spent: 5, max: 40, now, ...period })).toBeUndefined()
  })

  it("is now once the limit is used up", () => {
    const now = START + 24 * HOUR
    expect(runsOutAt({ spent: 40, max: 40, now, ...period })).toBe(now)
  })

  it("counts a fresh period as six hours, so one request is not a day's pace", () => {
    const now = START + HOUR
    // $1 in the first hour would be $24 a day; over six hours it is $4.
    expect(runsOutAt({ spent: 1, max: 40, now, ...period })).toBe(
      now + 39 * 6 * HOUR,
    )
  })

  it("has no pace with nothing spent or no period start", () => {
    expect(
      runsOutAt({ spent: 0, max: 40, now: START + HOUR, ...period }),
    ).toBeUndefined()
    expect(
      runsOutAt({
        spent: 10,
        max: 40,
        now: START + HOUR,
        periodStart: null,
        periodEnd: null,
      }),
    ).toBeUndefined()
  })
})
