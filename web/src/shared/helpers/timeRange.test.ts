import { describe, expect, it } from "vitest"

import {
  ACTIVITY_PRESETS,
  bucketDurationMs,
  bucketForWindow,
  bucketIndexRange,
  DAY_S,
  findPreset,
  formatWindowLabel,
  HOUR_S,
  isoAgo,
  rangeFromBuckets,
  USAGE_PRESETS,
} from "./timeRange"

describe("presets", () => {
  it("Usage presets are all bounded (no unbounded 'All')", () => {
    expect(USAGE_PRESETS.every((p) => typeof p.seconds === "number")).toBe(true)
    expect(USAGE_PRESETS.some((p) => p.key === "all")).toBe(false)
    expect(findPreset(USAGE_PRESETS, "90d")).toBeDefined()
    expect(findPreset(USAGE_PRESETS, "12mo")).toBeDefined()
  })

  it("Activity keeps a truthful unbounded 'All'", () => {
    expect(findPreset(ACTIVITY_PRESETS, "all")?.seconds).toBeNull()
  })
})

describe("isoAgo", () => {
  it("anchors to now minus the span", () => {
    const now = Date.parse("2026-07-27T12:00:00Z")
    expect(isoAgo(DAY_S, now)).toBe("2026-07-26T12:00:00.000Z")
  })
})

describe("bucketForWindow / bucketDurationMs", () => {
  const now = Date.parse("2026-07-27T00:00:00Z")

  it("buckets a sub-day (or exactly one-day) window hourly, longer daily", () => {
    expect(
      bucketForWindow("2026-07-26T00:00:00Z", "2026-07-27T00:00:00Z", now),
    ).toBe("hour")
    expect(
      bucketForWindow("2026-07-20T00:00:00Z", "2026-07-27T00:00:00Z", now),
    ).toBe("day")
  })

  it("measures an open-ended window against now", () => {
    expect(bucketForWindow("2026-07-26T23:00:00Z", undefined, now)).toBe("hour")
    expect(bucketForWindow("2026-07-01T00:00:00Z", undefined, now)).toBe("day")
  })

  it("reports the bucket duration", () => {
    expect(bucketDurationMs("hour")).toBe(HOUR_S * 1000)
    expect(bucketDurationMs("day")).toBe(DAY_S * 1000)
  })
})

describe("rangeFromBuckets", () => {
  const starts = [
    "2026-07-01T00:00:00Z",
    "2026-07-02T00:00:00Z",
    "2026-07-03T00:00:00Z",
    "2026-07-04T00:00:00Z",
  ]

  it("spans from the first picked bucket to the exclusive end of the last", () => {
    const r = rangeFromBuckets(starts, 1, 2, "day")
    expect(r).toEqual({
      startIso: "2026-07-02T00:00:00.000Z",
      endIso: "2026-07-04T00:00:00.000Z",
    })
  })

  it("is order-independent and clamps to the series bounds", () => {
    const r = rangeFromBuckets(starts, 9, -3, "day")
    expect(r).toEqual({
      startIso: "2026-07-01T00:00:00.000Z",
      endIso: "2026-07-05T00:00:00.000Z",
    })
  })

  it("returns null for an empty series", () => {
    expect(rangeFromBuckets([], 0, 0, "day")).toBeNull()
  })
})

describe("bucketIndexRange", () => {
  const starts = [
    "2026-07-01T00:00:00Z",
    "2026-07-02T00:00:00Z",
    "2026-07-03T00:00:00Z",
    "2026-07-04T00:00:00Z",
    "2026-07-05T00:00:00Z",
  ]

  it("spans the whole series when the window covers it (or is absent)", () => {
    expect(bucketIndexRange(starts, undefined, undefined)).toEqual({
      startIndex: 0,
      endIndex: 4,
    })
    expect(
      bucketIndexRange(starts, "2026-06-01T00:00:00Z", "2026-08-01T00:00:00Z"),
    ).toEqual({
      startIndex: 0,
      endIndex: 4,
    })
  })

  it("locates a sub-window by its bucket boundaries", () => {
    // Jul 2 .. Jul 4 exclusive -> buckets index 1 (Jul 2) through 2 (Jul 3).
    expect(
      bucketIndexRange(starts, "2026-07-02T00:00:00Z", "2026-07-04T00:00:00Z"),
    ).toEqual({
      startIndex: 1,
      endIndex: 2,
    })
  })

  it("clamps an inverted or empty window and handles an empty series", () => {
    const r = bucketIndexRange(
      starts,
      "2026-07-04T00:00:00Z",
      "2026-07-02T00:00:00Z",
    )
    expect(r.endIndex).toBeGreaterThanOrEqual(r.startIndex)
    expect(
      bucketIndexRange([], "2026-07-01T00:00:00Z", "2026-07-02T00:00:00Z"),
    ).toEqual({ startIndex: 0, endIndex: 0 })
  })
})

describe("formatWindowLabel", () => {
  it("labels an unbounded window", () => {
    expect(formatWindowLabel(undefined, undefined)).toBe("All time")
  })

  it("labels an open-ended (start-only) window", () => {
    expect(formatWindowLabel("2026-07-03T00:00:00Z", undefined)).toContain(
      "Since",
    )
    expect(formatWindowLabel("2026-07-03T00:00:00Z", undefined)).toContain(
      "Jul 3",
    )
  })

  it("shows the inclusive last day for a bounded range, not the exclusive bound", () => {
    const label = formatWindowLabel(
      "2026-07-03T00:00:00Z",
      "2026-07-11T00:00:00Z",
    )
    expect(label).toContain("Jul 3")
    expect(label).toContain("Jul 10")
    expect(label).not.toContain("Jul 11")
    expect(label).toContain("–")
  })

  it("collapses a single-day window to one date", () => {
    const label = formatWindowLabel(
      "2026-07-03T00:00:00Z",
      "2026-07-04T00:00:00Z",
    )
    expect(label).toContain("Jul 3")
    expect(label).not.toContain("–")
  })

  it("carries the year on both sides across a year boundary", () => {
    const label = formatWindowLabel(
      "2025-12-30T00:00:00Z",
      "2026-01-03T00:00:00Z",
    )
    expect(label).toContain("2025")
    expect(label).toContain("2026")
    expect(label).toContain("Jan 2")
  })
})
