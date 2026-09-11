import { act, renderHook } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { usePrefersReducedMotion } from "@/shared/hooks/usePrefersReducedMotion"

type Listener = (event: MediaQueryListEvent) => void

/** A `matchMedia` whose answer a test can change, the way a setting does. */
function stubMatchMedia(matches: boolean) {
  const listeners: Listener[] = []
  const query = {
    matches,
    addEventListener: (_: string, listener: Listener) =>
      listeners.push(listener),
    removeEventListener: (_: string, listener: Listener) => {
      listeners.splice(listeners.indexOf(listener), 1)
    },
  }
  vi.stubGlobal(
    "matchMedia",
    vi.fn(() => query),
  )
  return {
    change(next: boolean) {
      query.matches = next
      for (const listener of [...listeners]) {
        listener({ matches: next } as MediaQueryListEvent)
      }
    },
    get listenerCount() {
      return listeners.length
    },
  }
}

describe("usePrefersReducedMotion", () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it("reports the preference the system already holds", () => {
    stubMatchMedia(true)
    expect(renderHook(() => usePrefersReducedMotion()).result.current).toBe(
      true,
    )
  })

  it("follows a change made while the page is open", () => {
    // A system setting or a browser toggle can move under a page nobody
    // reloaded, and a canvas told to pause once would keep animating.
    const media = stubMatchMedia(false)
    const { result } = renderHook(() => usePrefersReducedMotion())
    expect(result.current).toBe(false)

    act(() => media.change(true))

    expect(result.current).toBe(true)
  })

  it("unsubscribes when it goes away", () => {
    const media = stubMatchMedia(false)
    const { unmount } = renderHook(() => usePrefersReducedMotion())
    expect(media.listenerCount).toBe(1)

    unmount()

    expect(media.listenerCount).toBe(0)
  })

  it("answers false where there is no matchMedia at all", () => {
    // Less motion is a preference somebody expressed, and an environment that
    // cannot be asked has not expressed it.
    vi.stubGlobal("matchMedia", undefined)
    expect(renderHook(() => usePrefersReducedMotion()).result.current).toBe(
      false,
    )
  })
})
