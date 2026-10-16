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
  it("defers to the latency format under a minute", () => {
    expect(formatDuration(850)).toBe("850 ms")
    expect(formatDuration(12_340)).toBe("12.34 s")
  })

  it("switches to minutes and padded seconds from a minute up", () => {
    expect(formatDuration(60_000)).toBe("1m 00s")
    expect(formatDuration(125_000)).toBe("2m 05s")
  })

  it("rounds to the second before splitting, so seconds never read 60", () => {
    expect(formatDuration(119_600)).toBe("2m 00s")
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
