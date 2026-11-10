import { describe, expect, it } from "vitest"

import { tickStep } from "@/features/traces/timelineAxis"

const MINUTE = 60_000
const HOUR = 60 * MINUTE
const DAY = 24 * HOUR

describe("tickStep", () => {
  it("steps by 1, 2 and 5 below a second", () => {
    expect(tickStep(900, 560, 56)).toBe(100)
    expect(tickStep(1_500, 560, 56)).toBe(200)
    expect(tickStep(40, 560, 56)).toBe(5)
  })

  it("steps by the divisions of a minute and an hour past that", () => {
    // 500 s over ten ticks is 50 s a tick, which rounds up to a minute.
    expect(tickStep(500_000, 560, 56)).toBe(MINUTE)
    expect(tickStep(100_000, 560, 56)).toBe(10_000)
    expect(tickStep(130_000, 560, 56)).toBe(15_000)
    expect(tickStep(50 * MINUTE, 560, 56)).toBe(5 * MINUTE)
    expect(tickStep(26 * HOUR, 560, 56)).toBe(3 * HOUR)
  })

  it("steps by whole days once a tick is longer than one", () => {
    expect(tickStep(30 * DAY, 560, 56)).toBe(5 * DAY)
    expect(tickStep(15 * DAY, 560, 56)).toBe(2 * DAY)
  })

  it("never steps by less than a millisecond, or by nothing", () => {
    expect(tickStep(1, 5_600, 56)).toBe(1)
    expect(tickStep(1, 10, 56)).toBe(1)
  })
})
