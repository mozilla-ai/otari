import { describe, expect, it } from "vitest"

import { defaultFaceFor, FACE_COUNT, face, sampleFaces } from "./faces"

/** The lit cells of a face, read back off its path, as "x,y" strings. */
function litCells(id: number): Set<string> {
  const lit = new Set<string>()
  const { sides, center } = face(id)
  for (const run of (sides + center).matchAll(/M(\d+) (\d+)h(\d+)/g)) {
    const [x, y, width] = [Number(run[1]), Number(run[2]), Number(run[3])]
    for (let i = 0; i < width; i++) lit.add(`${x + i},${y}`)
  }
  return lit
}

describe("face", () => {
  it("holds enough faces that a team rarely shares one", () => {
    // About 204,000: a team of 30 shares a face about 0.2% of the time.
    expect(FACE_COUNT).toBe(8 * 25_487)
  })

  it("walks tone fastest, so neighboring ids differ in color first", () => {
    expect(face(0).tone).toBe(1)
    expect(face(1).tone).toBe(2)
    expect(face(0).sides).toBe(face(1).sides)
    expect(face(0).center).toBe(face(1).center)
  })

  it("draws every face mirrored, inside the margin, with 6 to 10 cells lit", () => {
    for (let id = 0; id < FACE_COUNT; id += 997) {
      const lit = litCells(id)
      for (const cell of lit) {
        const [x, y] = cell.split(",").map(Number)
        expect(x).toBeGreaterThanOrEqual(1)
        expect(x).toBeLessThanOrEqual(5)
        expect(y).toBeGreaterThanOrEqual(1)
        expect(y).toBeLessThanOrEqual(5)
        expect(lit.has(`${6 - x},${y}`)).toBe(true)
      }
      const free = [...lit].filter((cell) => Number(cell.split(",")[0]) <= 3)
      expect(free.length).toBeGreaterThanOrEqual(6)
      expect(free.length).toBeLessThanOrEqual(10)
    }
  })

  it("names faces distinctly", () => {
    const labels = new Set(
      Array.from({ length: 4000 }, (_, id) => face(id).label),
    )
    expect(labels.size).toBe(4000)
  })

  it("draws the center column, and only it, in a second tone", () => {
    for (let id = 0; id < FACE_COUNT; id += 1499) {
      const { tone, accent, center } = face(id)
      expect(accent).not.toBe(tone)
      expect(center).not.toBe("")
      for (const run of center.matchAll(/M(\d+) /g)) expect(run[1]).toBe("3")
    }
  })

  it("wraps an out-of-range id rather than drawing nothing", () => {
    expect(face(FACE_COUNT).id).toBe(0)
    expect(face(-1).id).toBe(FACE_COUNT - 1)
  })
})

describe("defaultFaceFor", () => {
  it("gives the same id the same face every time", () => {
    const id = "33333333-3333-3333-3333-333333333333"
    expect(defaultFaceFor(id)).toBe(defaultFaceFor(id))
  })

  it("stays inside the set", () => {
    for (const seed of ["", "a", "ada@example.com", "z".repeat(500)]) {
      const id = defaultFaceFor(seed)
      expect(id).toBeGreaterThanOrEqual(0)
      expect(id).toBeLessThan(FACE_COUNT)
    }
  })

  it("gives near-identical ids different faces", () => {
    const faces = new Set(
      Array.from({ length: 200 }, (_, i) =>
        defaultFaceFor(
          `00000000-0000-0000-0000-${String(i).padStart(12, "0")}`,
        ),
      ),
    )
    // 200 people over ~204,000 faces: a sound hash all but never repeats.
    expect(faces.size).toBeGreaterThanOrEqual(198)
  })
})

describe("sampleFaces", () => {
  it("deals the number asked for, all distinct and inside the set", () => {
    const sample = sampleFaces(288)
    expect(new Set(sample).size).toBe(288)
    for (const id of sample) {
      expect(id).toBeGreaterThanOrEqual(0)
      expect(id).toBeLessThan(FACE_COUNT)
    }
  })
})
