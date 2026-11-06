import { describe, expect, it } from "vitest"

import { defaultFaceFor, FACE_COUNT, face } from "./faces"

describe("face", () => {
  it("walks tone fastest, so neighbouring ids differ in colour first", () => {
    expect(face(0).tone).toBe(1)
    expect(face(1).tone).toBe(2)
    expect(face(0).label.split(" ")[1]).toBe(face(1).label.split(" ")[1])
  })

  it("names every face distinctly", () => {
    const labels = new Set(
      Array.from({ length: FACE_COUNT }, (_, id) => face(id).label),
    )
    expect(labels.size).toBe(FACE_COUNT)
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

  it("spreads ids that share a prefix across the set", () => {
    const faces = new Set(
      Array.from({ length: 200 }, (_, i) =>
        defaultFaceFor(
          `00000000-0000-0000-0000-${String(i).padStart(12, "0")}`,
        ),
      ),
    )
    // 200 near-identical uuids over 72 faces: a weak hash would bunch them.
    expect(faces.size).toBeGreaterThan(50)
  })
})
