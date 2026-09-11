import { type RefObject, useEffect, useRef } from "react"
import { BAR_ROW_RATIO, type LoginBackgroundConfig } from "./config"
import { type BarGeometry, type BarPalette, drawBars } from "./renderBars"

export function LoginBackground({
  panelRef,
  config,
}: {
  panelRef: RefObject<HTMLDivElement | null>
  config: LoginBackgroundConfig
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const { columns, speed, intensity, contrast, spacing, rounding, waveScale } =
    config
  useEffect(() => {
    const settings = {
      columns,
      speed,
      intensity,
      contrast,
      spacing,
      rounding,
      waveScale,
    }
    const canvas = canvasRef.current
    const panel = panelRef.current
    const parent = canvas?.parentElement
    if (!canvas || !panel || !parent) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)")
    let geometry: BarGeometry
    let palette: BarPalette
    let frame = 0
    let lastFrame = 0
    let lastPaint = -Infinity
    let paintInterval = 1000 / 24
    let time = 0
    let geometryDirty = true
    let paletteDirty = true

    const canAnimate = () =>
      !document.hidden &&
      !reduced.matches &&
      geometry.width > 0 &&
      geometry.height > 0 &&
      speed > 0
    const paint = () => drawBars(ctx, geometry, palette, settings, time)
    const measure = () => {
      const bounds = parent.getBoundingClientRect()
      const anchor = panel.getBoundingClientRect()
      geometry = {
        width: bounds.width,
        height: bounds.height,
        left: anchor.left - bounds.left,
        top: anchor.top - bounds.top,
        panelWidth: anchor.width,
        panelHeight: anchor.height,
      }
      const pitch = anchor.width / columns
      // Preserve studio sizing; larger grids get fewer paints, at most 24,000 cells/second.
      const cells =
        pitch > 0
          ? (Math.ceil(bounds.width / pitch) + 1) *
            (Math.ceil(bounds.height / (pitch * BAR_ROW_RATIO)) + 1)
          : 0
      paintInterval = Math.max(1000 / 24, (cells * 1000) / 24000)
      const pixelRatio = Math.min(window.devicePixelRatio || 1, 1.5)
      const width = Math.round(bounds.width * pixelRatio)
      const height = Math.round(bounds.height * pixelRatio)
      if (canvas.width !== width) canvas.width = width
      if (canvas.height !== height) canvas.height = height
      ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0)
    }
    const readPalette = () => {
      const style = getComputedStyle(canvas)
      palette = {
        background: style.getPropertyValue("--color-background").trim(),
        accent: style.getPropertyValue("--color-primary").trim(),
      }
    }
    const animate = (now: number) => {
      frame = 0
      if (document.hidden) {
        lastFrame = 0
        return
      }
      const invalidated = geometryDirty || paletteDirty
      if (geometryDirty) measure()
      if (paletteDirty) readPalette()
      geometryDirty = false
      paletteDirty = false
      const moving = canAnimate()
      if (moving && lastFrame)
        time += Math.min((now - lastFrame) / 1000, 0.1) * speed
      lastFrame = moving ? now : 0
      if ((invalidated && !moving) || now - lastPaint >= paintInterval) {
        paint()
        lastPaint = now
      }
      if (moving) frame = requestAnimationFrame(animate)
    }
    const schedule = () => {
      if (!frame && !document.hidden) frame = requestAnimationFrame(animate)
    }
    const invalidateGeometry = () => {
      geometryDirty = true
      schedule()
    }
    const invalidatePalette = () => {
      paletteDirty = true
      schedule()
    }
    const update = () => {
      cancelAnimationFrame(frame)
      frame = 0
      lastFrame = 0
      invalidateGeometry()
    }
    schedule()
    const resize = new ResizeObserver(invalidateGeometry)
    resize.observe(parent)
    resize.observe(panel)
    const theme = new MutationObserver(invalidatePalette)
    theme.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class", "data-theme"],
    })
    reduced.addEventListener("change", update)
    document.addEventListener("visibilitychange", update)
    window.addEventListener("resize", invalidateGeometry)
    return () => {
      cancelAnimationFrame(frame)
      resize.disconnect()
      theme.disconnect()
      reduced.removeEventListener("change", update)
      document.removeEventListener("visibilitychange", update)
      window.removeEventListener("resize", invalidateGeometry)
    }
  }, [
    panelRef,
    columns,
    speed,
    intensity,
    contrast,
    spacing,
    rounding,
    waveScale,
  ])

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      tabIndex={-1}
      className="pointer-events-none absolute inset-0 h-full w-full"
    />
  )
}
