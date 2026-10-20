import { describe, expect, it } from "vitest"

import {
  formatDuration,
  formatRelativeTime,
  formatTraceTimestamp,
} from "@/features/traces/format"

describe("formatTraceTimestamp", () => {
  // Built from local parts, because the format is local time by design.
  const iso = new Date(2026, 0, 5, 7, 8, 9, 42).toISOString()

  it("pads every field to a fixed width", () => {
    expect(formatTraceTimestamp(iso)).toBe("2026-01-05 07:08:09")
  })

  it("adds the milliseconds, padded to three digits, when asked", () => {
    expect(formatTraceTimestamp(iso, true)).toBe("2026-01-05 07:08:09.042")
  })
})

describe("formatDuration", () => {
  it("keeps the latency format under a minute", () => {
    expect(formatDuration(850)).toBe("850 ms")
    expect(formatDuration(21_310)).toBe("21.31 s")
  })

  it("counts minutes and seconds under an hour", () => {
    expect(formatDuration(60_000)).toBe("1m 00s")
    expect(formatDuration(125_000)).toBe("2m 05s")
    expect(formatDuration(59 * 60_000 + 59_000)).toBe("59m 59s")
  })

  it("rounds to the second before splitting, so seconds never read 60", () => {
    expect(formatDuration(119_600)).toBe("2m 00s")
  })

  it("counts hours and minutes from an hour up, so a long session stays short", () => {
    expect(formatDuration(3_600_000)).toBe("1h 00m")
    expect(formatDuration(3_599_600)).toBe("1h 00m")
    expect(formatDuration((1325 * 60 + 40) * 1000)).toBe("22h 06m")
  })
})

describe("formatRelativeTime", () => {
  it("reads minutes and padded seconds from the trace start", () => {
    expect(formatRelativeTime(0)).toBe("0:00")
    expect(formatRelativeTime(65_999)).toBe("1:05")
  })

  it("appends padded milliseconds when asked, truncating a fraction", () => {
    expect(formatRelativeTime(65_007.9, true)).toBe("1:05.007")
  })
})
