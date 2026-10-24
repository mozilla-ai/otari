import { describe, expect, it } from "vitest"

import { type Rect, squarify } from "./squarify"

const BOUNDS: Rect = { x: 0, y: 0, width: 640, height: 320 }

function area(rect: Rect) {
  return rect.width * rect.height
}

function overlap(a: Rect, b: Rect) {
  const width = Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x)
  const height = Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y)
  return width > 1e-6 && height > 1e-6 ? width * height : 0
}

const SETS = [
  [50],
  [60, 40],
  [39_600, 160_400],
  [14_800, 12_000, 9_000, 3_000, 800, 200, 5],
  [1, 1, 1, 1, 1, 1, 1],
  [1_000_000, 1],
]

describe("squarify", () => {
  it.each(SETS)(
    "gives each item an area in proportion to its value (%j)",
    (...values) => {
      const placed = squarify(
        values.map((value) => ({ value })),
        BOUNDS,
      )
      const total = values.reduce((sum, value) => sum + value, 0)

      expect(placed).toHaveLength(values.length)
      for (const { item, ...rect } of placed) {
        expect(area(rect) / area(BOUNDS)).toBeCloseTo(item.value / total, 9)
      }
    },
  )

  it.each(SETS)(
    "tiles the bounds without overlap or a gap (%j)",
    (...values) => {
      const placed = squarify(
        values.map((value) => ({ value })),
        BOUNDS,
      )

      expect(placed.reduce((sum, rect) => sum + area(rect), 0)).toBeCloseTo(
        area(BOUNDS),
        6,
      )
      for (const rect of placed) {
        expect(rect.x).toBeGreaterThanOrEqual(-1e-9)
        expect(rect.y).toBeGreaterThanOrEqual(-1e-9)
        expect(rect.x + rect.width).toBeLessThanOrEqual(BOUNDS.width + 1e-6)
        expect(rect.y + rect.height).toBeLessThanOrEqual(BOUNDS.height + 1e-6)
      }
      for (let i = 0; i < placed.length; i += 1)
        for (let j = i + 1; j < placed.length; j += 1)
          expect(overlap(placed[i], placed[j])).toBe(0)
    },
  )

  it("lays out inside bounds that do not start at the origin", () => {
    const bounds = { x: 10, y: 20, width: 100, height: 50 }
    const placed = squarify([{ value: 3 }, { value: 1 }], bounds)

    expect(Math.min(...placed.map((rect) => rect.x))).toBeCloseTo(10)
    expect(Math.min(...placed.map((rect) => rect.y))).toBeCloseTo(20)
    expect(placed.reduce((sum, rect) => sum + area(rect), 0)).toBeCloseTo(5000)
  })

  it("puts the largest item first and keeps the order of equal ones", () => {
    const placed = squarify(
      [
        { value: 1, name: "a" },
        { value: 5, name: "b" },
        { value: 1, name: "c" },
      ],
      BOUNDS,
    )

    expect(placed.map((rect) => rect.item.name)).toEqual(["b", "a", "c"])
  })

  it("keeps the shapes close to square", () => {
    const placed = squarify(
      [30, 25, 20, 10, 8, 4, 3].map((value) => ({ value })),
      { x: 0, y: 0, width: 600, height: 400 },
    )

    for (const rect of placed) {
      expect(
        Math.max(rect.width / rect.height, rect.height / rect.width),
      ).toBeLessThan(4)
    }
  })

  it("leaves out an item with no value, and everything for empty bounds", () => {
    expect(
      squarify(
        [{ value: 0 }, { value: -3 }, { value: Number.NaN }, { value: 2 }],
        BOUNDS,
      ),
    ).toHaveLength(1)
    expect(
      squarify([{ value: 1 }], { x: 0, y: 0, width: 0, height: 10 }),
    ).toEqual([])
    expect(squarify([], BOUNDS)).toEqual([])
  })
})
