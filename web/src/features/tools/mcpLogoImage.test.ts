import { describe, expect, it } from "vitest"

import { inkLuminance, logoTone } from "@/features/tools/mcpLogoImage"

function square(
  size: number,
  paint: (x: number, y: number) => [number, number, number, number],
): Uint8ClampedArray {
  const pixels = new Uint8ClampedArray(size * size * 4)
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      pixels.set(paint(x, y), (y * size + x) * 4)
    }
  }
  return pixels
}

describe("inkLuminance", () => {
  it("weighs the ink by its opacity and ignores the transparent field", () => {
    const mark = square(8, (x, y) =>
      x > 1 && x < 6 && y > 1 && y < 6 ? [255, 255, 255, 255] : [0, 0, 0, 0],
    )
    expect(inkLuminance(mark, 8)).toBeCloseTo(1)
  })

  it("gives a logo on its own filled square no luminance, so no tile", () => {
    expect(
      inkLuminance(
        square(8, () => [0, 0, 0, 255]),
        8,
      ),
    ).toBeUndefined()
  })
})

describe("logoTone", () => {
  it("tiles dark ink, light ink, always, or never as the gateway does", () => {
    expect(logoTone("auto", 0.1)).toBe("dark")
    expect(logoTone("auto", 0.95)).toBe("light")
    expect(logoTone("auto", 0.5)).toBeUndefined()
    expect(logoTone("auto", undefined)).toBeUndefined()
    expect(logoTone("always", 0.5)).toBe("tile")
    expect(logoTone("never", 0.1)).toBeUndefined()
  })
})
