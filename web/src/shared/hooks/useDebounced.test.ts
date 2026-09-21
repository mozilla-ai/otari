import { act, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { useDebounced } from "@/shared/hooks/useDebounced"

describe("useDebounced", () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it("returns the first value without waiting", () => {
    // A field that mounts with a term should not sit empty for the delay.
    const { result } = renderHook(() => useDebounced("gpt", 250))

    expect(result.current).toBe("gpt")
  })

  it("holds a change until the typing stops", () => {
    const { result, rerender } = renderHook(
      ({ value }) => useDebounced(value, 250),
      { initialProps: { value: "" } },
    )

    rerender({ value: "g" })
    expect(result.current).toBe("")

    act(() => {
      vi.advanceTimersByTime(250)
    })
    expect(result.current).toBe("g")
  })

  it("reports only the last of a burst", () => {
    // The point: one request for "gpt" rather than three whose answers race.
    const { result, rerender } = renderHook(
      ({ value }) => useDebounced(value, 250),
      { initialProps: { value: "" } },
    )

    rerender({ value: "g" })
    act(() => {
      vi.advanceTimersByTime(100)
    })
    rerender({ value: "gp" })
    act(() => {
      vi.advanceTimersByTime(100)
    })
    rerender({ value: "gpt" })
    expect(result.current).toBe("")

    act(() => {
      vi.advanceTimersByTime(250)
    })
    expect(result.current).toBe("gpt")
  })

  it("clears its timer on unmount", () => {
    const { rerender, unmount } = renderHook(
      ({ value }) => useDebounced(value, 250),
      { initialProps: { value: "" } },
    )

    rerender({ value: "g" })
    unmount()

    // Nothing left to fire: a timer outliving its component sets state on one
    // that is gone.
    expect(vi.getTimerCount()).toBe(0)
  })
})
