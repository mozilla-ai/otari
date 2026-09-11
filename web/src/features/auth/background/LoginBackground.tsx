import { type RefObject, useEffect, useRef } from "react"
import type { LoginBackgroundConfig } from "./config"
import { type BarGeometry, type BarPalette, drawBars } from "./renderBars"

export function LoginBackground({
  panelRef,
  config,
}: {
  panelRef: RefObject<HTMLDivElement | null>
  config: LoginBackgroundConfig
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
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
    let lastPaint = 0
    let time = 0
    let geometryDirty = true
    let paletteDirty = true

    const canAnimate = () =>
      !document.hidden &&
      !reduced.matches &&
      geometry.visibleBottom > geometry.visibleTop &&
      config.speed > 0
    const paint = () => drawBars(ctx, geometry, palette, config, time)
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
        visibleTop: Math.max(0, -bounds.top),
        visibleBottom: Math.min(bounds.height, window.innerHeight - bounds.top),
      }
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
        time += Math.min((now - lastFrame) / 1000, 0.1) * config.speed
      lastFrame = moving ? now : 0
      // Invalidation and animation share a frame, including the initial resize observation.
      if (invalidated || now - lastPaint >= 1000 / 24) {
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
    document.addEventListener("scroll", invalidateGeometry, {
      passive: true,
      capture: true,
    })
    window.addEventListener("resize", invalidateGeometry)
    return () => {
      cancelAnimationFrame(frame)
      resize.disconnect()
      theme.disconnect()
      reduced.removeEventListener("change", update)
      document.removeEventListener("visibilitychange", update)
      document.removeEventListener("scroll", invalidateGeometry, true)
      window.removeEventListener("resize", invalidateGeometry)
    }
  }, [panelRef, config])

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      tabIndex={-1}
      className="pointer-events-none absolute inset-0 h-full w-full"
    />
  )
}
