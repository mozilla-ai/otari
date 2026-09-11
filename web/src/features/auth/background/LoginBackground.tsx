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

    const canAnimate = () =>
      !document.hidden &&
      !reduced.matches &&
      geometry.visibleBottom > geometry.visibleTop &&
      config.speed > 0
    const paint = () => drawBars(ctx, geometry, palette, config, time)
    const animate = (now: number) => {
      frame = 0
      if (!canAnimate()) {
        lastFrame = 0
        return
      }
      if (lastFrame)
        time += Math.min((now - lastFrame) / 1000, 0.1) * config.speed
      lastFrame = now
      // The slow decorative field needs at most 24 paints per second.
      if (now - lastPaint >= 1000 / 24) {
        paint()
        lastPaint = now
      }
      frame = requestAnimationFrame(animate)
    }
    const update = () => {
      paint()
      if (canAnimate()) {
        if (!frame) frame = requestAnimationFrame(animate)
      } else {
        cancelAnimationFrame(frame)
        frame = 0
        lastFrame = 0
      }
    }
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
      const style = getComputedStyle(canvas)
      palette = {
        background: style.getPropertyValue("--color-background").trim(),
        accent: style.getPropertyValue("--color-primary").trim(),
      }
      update()
    }
    measure()
    const resize = new ResizeObserver(measure)
    resize.observe(parent)
    resize.observe(panel)
    const theme = new MutationObserver(measure)
    theme.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class", "data-theme"],
    })
    reduced.addEventListener("change", update)
    document.addEventListener("visibilitychange", update)
    document.addEventListener("scroll", measure, {
      passive: true,
      capture: true,
    })
    window.addEventListener("resize", measure)
    return () => {
      cancelAnimationFrame(frame)
      resize.disconnect()
      theme.disconnect()
      reduced.removeEventListener("change", update)
      document.removeEventListener("visibilitychange", update)
      document.removeEventListener("scroll", measure, true)
      window.removeEventListener("resize", measure)
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
