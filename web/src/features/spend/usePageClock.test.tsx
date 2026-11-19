import { act, renderHook } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { usePageClock } from "./usePageClock"

afterEach(() => {
  vi.useRealTimers()
})

describe("usePageClock", () => {
  it("holds still between renders and moves on when asked", () => {
    vi.useFakeTimers({ now: new Date("2026-11-18T09:00:00Z") })
    const { result, rerender } = renderHook(() => usePageClock())
    const opened = result.current[0]
    vi.setSystemTime(new Date("2026-11-18T17:00:00Z"))
    rerender()
    expect(result.current[0]).toBe(opened)
    act(() => result.current[1]())
    expect(result.current[0]).toBe(Date.parse("2026-11-18T17:00:00Z"))
  })

  it("moves on when the tab comes back into view", () => {
    vi.useFakeTimers({ now: new Date("2026-11-18T23:50:00Z") })
    const { result } = renderHook(() => usePageClock())
    vi.setSystemTime(new Date("2026-11-19T08:00:00Z"))
    act(() => {
      document.dispatchEvent(new Event("visibilitychange"))
    })
    expect(result.current[0]).toBe(Date.parse("2026-11-19T08:00:00Z"))
  })
})
