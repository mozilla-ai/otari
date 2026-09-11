import { type RefObject, useEffect, useRef } from "react"
import type { LoginBackgroundConfig } from "./config"
import { type BarGeometry, type BarPalette, drawBars } from "./renderBars"

export function LoginBackground({
  panelRef,
  config,
  paused,
}: {
  panelRef: RefObject<HTMLDivElement | null>
  config: LoginBackgroundConfig
  paused: boolean
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const settings = useRef({ config, paused })
  const redraw = useRef<(() => void) | undefined>(undefined)

  useEffect(() => {
    settings.current = { config, paused }
    redraw.current?.()
  }, [config, paused])

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
    let time = 0

    const canAnimate = () =>
      !document.hidden &&
      !reduced.matches &&
      !settings.current.paused &&
      settings.current.config.speed > 0
    const paint = () =>
      drawBars(ctx, geometry, palette, settings.current.config, time)
    const animate = (now: number) => {
      frame = 0
      if (!canAnimate()) {
        lastFrame = 0
        return
      }
      if (lastFrame)
        time +=
          Math.min((now - lastFrame) / 1000, 0.1) *
          settings.current.config.speed
      lastFrame = now
      paint()
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
      }
      const pixelRatio = Math.min(window.devicePixelRatio || 1, 1.5)
      canvas.width = Math.round(bounds.width * pixelRatio)
      canvas.height = Math.round(bounds.height * pixelRatio)
      ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0)
      const style = getComputedStyle(canvas)
      palette = {
        background: style.getPropertyValue("--color-background").trim(),
        accent: style.getPropertyValue("--color-primary").trim(),
        foreground: style.getPropertyValue("--color-text").trim(),
      }
      update()
    }
    measure()
    redraw.current = update
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
    return () => {
      cancelAnimationFrame(frame)
      redraw.current = undefined
      resize.disconnect()
      theme.disconnect()
      reduced.removeEventListener("change", update)
      document.removeEventListener("visibilitychange", update)
    }
  }, [panelRef])

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      tabIndex={-1}
      className="pointer-events-none absolute inset-0 h-full w-full"
    />
  )
}
