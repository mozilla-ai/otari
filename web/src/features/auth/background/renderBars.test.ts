import { describe, expect, it, vi } from "vitest"
import { barLuminance } from "./barWave"
import { BAR_ROW_RATIO } from "./config"
import saved from "./login-background.json"
import { type BarGeometry, drawBars } from "./renderBars"

const palette = {
  background: "Canvas",
  accent: "Highlight",
}
const geometry: BarGeometry = {
  width: 800,
  height: 1200,
  left: 200,
  top: 200,
  panelWidth: 400,
  panelHeight: 400,
}

function context() {
  const fills: { alpha: number; color: string }[] = []
  const ctx = {
    globalAlpha: 1,
    fillStyle: "",
    clearRect: vi.fn(),
    fillRect: vi.fn(),
    beginPath: vi.fn(),
    roundRect: vi.fn(),
    fill() {
      fills.push({ alpha: this.globalAlpha, color: this.fillStyle })
    },
  }
  return { ctx, fills, canvas: ctx as unknown as CanvasRenderingContext2D }
}

describe("bar renderer", () => {
  it("paints the full tall canvas and skips cells hidden by the form", () => {
    const { ctx, fills, canvas } = context()
    drawBars(canvas, geometry, palette, saved, 0)
    expect(fills.length).toBeGreaterThan(0)
    expect(ctx.clearRect).not.toHaveBeenCalled()
    expect(ctx.fillRect).toHaveBeenCalledExactlyOnceWith(0, 0, 800, 1200)
    expect(ctx.roundRect.mock.calls.some(([, y]) => y > 1000)).toBe(true)
    for (const [x, y, width, height] of ctx.roundRect.mock.calls) {
      expect(y + height).toBeGreaterThan(0)
      expect(y).toBeLessThan(geometry.height)
      expect(x + width).toBeGreaterThan(0)
      expect(x).toBeLessThan(geometry.width)
      expect(
        x >= 200 && x + width <= 600 && y >= 200 && y + height <= 600,
      ).toBe(false)
    }
    expect(ctx.globalAlpha).toBe(1)
  })

  it("applies the configured wave contrast, intensity, and theme accent", () => {
    const { ctx, fills, canvas } = context()
    drawBars(canvas, geometry, palette, saved, 3)
    const pitchX = geometry.panelWidth / saved.columns
    const pitchY = pitchX * BAR_ROW_RATIO
    const [x, y, width, height] = ctx.roundRect.mock.calls[0]
    const luminance = barLuminance(
      (x + width / 2) / geometry.width,
      (y + height / 2) / geometry.height,
      3,
      saved.waveScale,
    )
    expect(pitchY).toBeGreaterThan(pitchX)
    expect(fills[0].alpha).toBeCloseTo(
      luminance ** saved.contrast * saved.intensity,
    )
    for (const fill of fills) {
      expect(fill.alpha).toBeGreaterThanOrEqual(0)
      expect(fill.alpha).toBeLessThanOrEqual(1)
      expect(fill.color).toBe("Highlight")
    }
  })

  it.each([390, 1440, 2560, 7680])(
    "preserves studio bar dimensions at viewport width %i",
    (width) => {
      const { ctx, canvas } = context()
      const panelWidth = Math.min(448, width - 32)
      drawBars(canvas, { ...geometry, width, panelWidth }, palette, saved, 0)
      const pitch = panelWidth / 20
      expect(ctx.roundRect.mock.calls.length).toBeGreaterThan(0)
      for (const [, , barWidth, barHeight, radius] of ctx.roundRect.mock
        .calls) {
        expect(barWidth).toBeCloseTo(pitch * 0.95)
        expect(barHeight).toBeCloseTo(
          pitch * BAR_ROW_RATIO * (1 - 0.05 * (0.134 / 0.248)),
        )
        expect(radius).toBeCloseTo(barWidth * 0.11)
      }
    },
  )

  it("does not paint empty geometry", () => {
    for (const changes of [{ height: 0 }, { panelWidth: 0 }]) {
      const { ctx, canvas } = context()
      drawBars(canvas, { ...geometry, ...changes }, palette, saved, 0)
      expect(ctx.clearRect).not.toHaveBeenCalled()
      expect(ctx.fillRect).not.toHaveBeenCalled()
      expect(ctx.roundRect).not.toHaveBeenCalled()
    }
  })
})
