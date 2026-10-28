import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import {
  BOOT_MARKS,
  markAfterPaint,
  markOnce,
  readBootTiming,
  whenMarked,
} from "@/shared/telemetry/bootTiming"

function marks(name: string): number {
  return performance.getEntriesByName(name, "mark").length
}

describe("boot timing marks", () => {
  beforeEach(() => {
    performance.clearMarks()
  })

  afterEach(() => {
    vi.restoreAllMocks()
    vi.useRealTimers()
  })

  it("writes each mark once per load", () => {
    markOnce(BOOT_MARKS.CONTEXT)
    markOnce(BOOT_MARKS.CONTEXT)
    expect(marks(BOOT_MARKS.CONTEXT)).toBe(1)
  })

  it("marks after the second frame, not the first", () => {
    const frames: FrameRequestCallback[] = []
    vi.spyOn(window, "requestAnimationFrame").mockImplementation((frame) => {
      frames.push(frame)
      return frames.length
    })

    markAfterPaint(BOOT_MARKS.SHELL)
    frames.shift()?.(0)
    expect(marks(BOOT_MARKS.SHELL)).toBe(0)
    frames.shift()?.(0)
    expect(marks(BOOT_MARKS.SHELL)).toBe(1)
  })

  it("tells a waiting listener once, and a late one straight away", async () => {
    const early = vi.fn()
    whenMarked(BOOT_MARKS.KPIS, early)
    markOnce(BOOT_MARKS.KPIS)
    markOnce(BOOT_MARKS.KPIS)
    expect(early).toHaveBeenCalledTimes(1)

    const late = vi.fn()
    whenMarked(BOOT_MARKS.KPIS, late)
    await Promise.resolve()
    expect(late).toHaveBeenCalledTimes(1)
  })

  it("does not tell a listener that stopped waiting", () => {
    const listener = vi.fn()
    const stop = whenMarked(BOOT_MARKS.KPIS, listener)
    stop()
    markOnce(BOOT_MARKS.KPIS)
    expect(listener).not.toHaveBeenCalled()
  })

  it("reads the kept answers as off when nothing kept them", () => {
    expect(readBootTiming()).toMatchObject({
      kept_bootstrap: "off",
      kept_context: "off",
      remounted: false,
    })
  })

  it("reads the kept outcomes and a remount where a build writes them", () => {
    performance.mark("boot:kept-bootstrap:hit")
    performance.mark("ctx:kept:miss")
    performance.mark(BOOT_MARKS.REMOUNT)
    expect(readBootTiming()).toMatchObject({
      kept_bootstrap: "hit",
      kept_context: "miss",
      remounted: true,
    })
  })

  it("ignores a kept mark whose outcome it does not know", () => {
    performance.mark("boot:kept-bootstrap:maybe")
    expect(readBootTiming().kept_bootstrap).toBe("off")
  })

  it("rounds a duration to 10 ms and caps it at a minute", () => {
    performance.mark(BOOT_MARKS.BOOTSTRAP, { startTime: 1234.4 })
    performance.mark(BOOT_MARKS.SHELL, { startTime: 1236 })
    performance.mark(BOOT_MARKS.KPIS, { startTime: 95_000 })
    const timing = readBootTiming()
    expect(timing.t_bootstrap_ms).toBe(1230)
    expect(timing.t_shell_ms).toBe(1240)
    expect(timing.t_kpi_ms).toBe(60_000)
    expect(timing.t_context_ms).toBeUndefined()
  })

  it("leaves out client hints the browser does not offer", () => {
    const timing = readBootTiming()
    expect(timing.device).toBeUndefined()
    expect(timing.effective_type).toBeUndefined()
  })

  it("reads the client hints where the browser offers them", () => {
    Object.defineProperty(navigator, "userAgentData", {
      configurable: true,
      value: { mobile: true },
    })
    Object.defineProperty(navigator, "connection", {
      configurable: true,
      value: { effectiveType: "3g" },
    })
    try {
      expect(readBootTiming()).toMatchObject({
        device: "mobile",
        effective_type: "3g",
      })
    } finally {
      Reflect.deleteProperty(navigator, "userAgentData")
      Reflect.deleteProperty(navigator, "connection")
    }
  })
})
