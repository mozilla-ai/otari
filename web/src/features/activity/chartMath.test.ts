import { describe, expect, it } from "vitest"
import {
  barBox,
  barCategoryGap,
  minPointSize,
  peakRange,
  pointedReading,
  sumRange,
  unitFor,
  windowReading,
  xLabelWidth,
  xTickIndices,
  yStep,
  yTicks,
} from "./chartMath"

describe("y step", () => {
  it("takes the smallest of the allowed steps that reaches max / (ticks - 1)", () => {
    // 329 over two intervals is 164.5: 150 is short, 200 is the next step.
    expect(yStep(329, 3)).toBe(200)
    expect(yStep(401, 3)).toBe(250)
    expect(yStep(100, 3)).toBe(50)
  })

  it("rounds up to an integer, since these are counts", () => {
    expect(yStep(1, 3)).toBe(1)
    expect(yStep(3, 3)).toBe(2)
  })

  it("lays out the ticks from zero to the top of the scale", () => {
    expect(yTicks(329, 3)).toEqual({ ticks: [0, 200, 400], top: 400 })
  })

  it("never returns a step of zero", () => {
    expect(yStep(0, 3)).toBe(1)
  })
})

describe("x ticks", () => {
  it("counts back from the latest, so the latest is always labelled", () => {
    // 31 buckets at 35.1px each: every third, 72px apart at least.
    const indices = xTickIndices(31, 35.1, 72)
    expect(indices[indices.length - 1]).toBe(30)
    expect(indices[0]).toBe(0)
    expect(indices).toHaveLength(11)
    expect(indices.slice(-3)).toEqual([24, 27, 30])
  })

  it("labels every bucket when each is wide enough", () => {
    expect(xTickIndices(8, 136, 72)).toEqual([0, 1, 2, 3, 4, 5, 6, 7])
  })

  it("steps by ceil(72 / pitch) and ends on the latest", () => {
    expect(xTickIndices(10, 30, 72)).toEqual([0, 3, 6, 9])
  })

  // A week of 6-hour buckets, labelled "Nov 10, 12:00 AM": every label is
  // centered on its bucket but the latest, which ends at the plot's right edge.
  it.each([
    ["a phone", 326, 64, 32],
    ["a desktop", 1100, 72, 40],
  ])(
    "never draws two week labels over each other on %s",
    (_, plot, minGap, gutter) => {
      const count = 28
      const pitch = plot / count
      const width = xLabelWidth(["Nov 10, 12:00 AM", "Nov 9, 6:00 PM"])
      const boxes = xTickIndices(count, pitch, minGap, width, gutter).map(
        (index) => {
          const right =
            index === count - 1 ? plot : (index + 0.5) * pitch + width / 2
          return { left: right - width, right }
        },
      )
      expect(boxes.length).toBeGreaterThan(1)
      expect(boxes[0].left).toBeGreaterThanOrEqual(-gutter)
      for (let i = 1; i < boxes.length; i += 1) {
        expect(boxes[i].left).toBeGreaterThanOrEqual(boxes[i - 1].right)
      }
    },
  )
})

describe("the readout's unit", () => {
  it("is singular for one", () => {
    expect(unitFor(1, "requests")).toBe("request")
    expect(unitFor(1, "active sessions")).toBe("active session")
  })

  it("is plural otherwise", () => {
    expect(unitFor(0, "requests")).toBe("requests")
    expect(unitFor(2, "active sessions")).toBe("active sessions")
  })
})

describe("bars", () => {
  it("draws the bar sizes the design names, at 1088px wide", () => {
    const size = (count: number) => {
      const pitch = 1088 / count
      return barBox(pitch, barCategoryGap(pitch)).size
    }
    expect(size(31)).toBe(26)
    expect(size(8)).toBe(40)
    expect(size(60)).toBe(14)
  })

  it("lets bars touch once a bucket is under 4px", () => {
    const pitch = 1088 / 366
    expect(barCategoryGap(pitch)).toBe(0)
    expect(barBox(pitch, 0).size).toBe(3)
    expect(barBox(pitch, 0).offset).toBe(0)
  })

  it("leaves a 9px gap at 31 buckets", () => {
    const pitch = 1088 / 31
    const { size } = barBox(pitch, barCategoryGap(pitch))
    expect(Math.round(pitch - size)).toBe(9)
  })

  it("gives any non-zero value at least two pixels", () => {
    expect(minPointSize(1)).toBe(2)
    expect(minPointSize(0)).toBe(0)
    expect(minPointSize(null)).toBe(0)
  })
})

const SERIES = [
  { requests: 10, errors: 1 },
  { requests: 20, errors: 0 },
  { requests: 5, errors: 9 },
  { requests: 0, errors: 0 },
]

describe("totals", () => {
  it("sums both series over an inclusive range, a failure never above its bucket", () => {
    expect(sumRange(SERIES, 0, 2)).toEqual({ count: 35, failed: 6 })
    expect(sumRange(SERIES, 2, 0)).toEqual({ count: 35, failed: 6 })
    expect(sumRange(SERIES, 1, 1)).toEqual({ count: 20, failed: 0 })
  })

  it("clamps a range to the series", () => {
    expect(sumRange(SERIES, -3, 99)).toEqual({ count: 35, failed: 6 })
  })

  it("takes the tallest bucket where buckets overlap", () => {
    expect(peakRange(SERIES, 0, 3)).toEqual({ count: 20, failed: 5 })
  })

  it("sums the window when the caller gives no totals and the series is additive", () => {
    expect(
      windowReading({
        totals: undefined,
        additive: true,
        loading: false,
        series: SERIES,
        startIndex: 0,
        endIndex: 1,
      }),
    ).toEqual({ count: 30, failed: 1 })
  })

  it("prefers the caller's totals, and never sums a series that is not additive", () => {
    expect(
      windowReading({
        totals: { count: 736, failed: null },
        additive: false,
        loading: false,
        series: SERIES,
        startIndex: 0,
        endIndex: 3,
      }),
    ).toEqual({ count: 736, failed: null })
    expect(
      windowReading({
        totals: undefined,
        additive: false,
        loading: false,
        series: SERIES,
        startIndex: 0,
        endIndex: 3,
      }),
    ).toMatchObject({ count: 20, peak: true })
  })

  it("is still loading when the caller says null, or when nothing has arrived", () => {
    expect(
      windowReading({
        totals: null,
        additive: true,
        loading: false,
        series: SERIES,
        startIndex: 0,
        endIndex: 3,
      }),
    ).toBeNull()
    expect(
      windowReading({
        totals: undefined,
        additive: true,
        loading: true,
        series: [],
        startIndex: 0,
        endIndex: 0,
      }),
    ).toBeNull()
  })

  it("reads one bucket as itself and a range as a sum, or a peak when not additive", () => {
    const base = { count: 35, failed: 6 }
    expect(
      pointedReading({ base, additive: true, series: SERIES, lo: 2, hi: 2 }),
    ).toEqual({ count: 5, failed: 5, peak: false })
    expect(
      pointedReading({ base, additive: true, series: SERIES, lo: 0, hi: 2 }),
    ).toEqual({ count: 35, failed: 6, peak: false })
    expect(
      pointedReading({ base, additive: false, series: SERIES, lo: 0, hi: 2 }),
    ).toEqual({ count: 20, failed: 5, peak: true })
    expect(
      pointedReading({ base, additive: false, series: SERIES, lo: 1, hi: 1 }),
    ).toEqual({ count: 20, failed: 0, peak: false })
  })

  it("keeps a chart without a failed figure without one under the pointer", () => {
    const base = { count: 736, failed: null }
    expect(
      pointedReading({ base, additive: false, series: SERIES, lo: 0, hi: 0 }),
    ).toMatchObject({ failed: null })
    expect(
      pointedReading({
        base: null,
        additive: true,
        series: SERIES,
        lo: 0,
        hi: 0,
      }),
    ).toBeNull()
  })
})
