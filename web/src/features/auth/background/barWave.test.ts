import { describe, expect, it } from "vitest"
import { barLuminance, WAVE_PERIOD } from "./barWave"

describe("continuous bar waves", () => {
  it("returns to the same position and velocity across its natural period", () => {
    const step = 0.0001
    for (let row = 0; row < 9; row++) {
      for (let column = 0; column < 55; column++) {
        const x = column / 54
        const y = (row + 0.5) / 9
        const start = barLuminance(x, y, 0)
        expect(barLuminance(x, y, WAVE_PERIOD)).toBeCloseTo(start, 10)
        const velocityBefore =
          (start - barLuminance(x, y, WAVE_PERIOD - step)) / step
        const velocityAfter = (barLuminance(x, y, step) - start) / step
        expect(velocityBefore).toBeCloseTo(velocityAfter, 2)
      }
    }
  })

  it("stays finite and bounded across the field and wave-size limits", () => {
    for (const scale of [0.5, 1, 2]) {
      for (let time = 0; time < WAVE_PERIOD; time += 0.1) {
        for (const y of [-0.1, 0, 0.25, 0.5, 0.75, 1, 1.1]) {
          const value = barLuminance(0.23, y, time, scale)
          expect(value).toBeGreaterThanOrEqual(0.06)
          expect(value).toBeLessThanOrEqual(0.92)
        }
      }
    }
  })

  it("samples subcell positions directly and does not repeat the short recording", () => {
    const at = (x: number, time: number) => barLuminance(x, 0.5, time)
    expect(at(0.5, 3.33333)).not.toBeCloseTo(at(0.5, 0), 2)
    const midpoint = at(0.5, 2)
    const neighboringAverage = (at(0.49, 2) + at(0.51, 2)) / 2
    expect(midpoint).not.toBeCloseTo(neighboringAverage, 6)
    expect(at(0.500001, 2)).toBeCloseTo(midpoint, 4)
  })
})
