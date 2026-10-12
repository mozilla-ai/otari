import { act, renderHook } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { useIsMobile } from "./useIsMobile"

const realMatchMedia = window.matchMedia

function stubMedia(matches: boolean) {
  let listener: (() => void) | undefined
  const query = {
    matches,
    media: "",
    addEventListener: (_: string, fn: () => void) => {
      listener = fn
    },
    removeEventListener: vi.fn(() => {
      listener = undefined
    }),
  }
  const matchMedia = vi.fn(() => query)
  window.matchMedia = matchMedia as unknown as typeof window.matchMedia
  return {
    matchMedia,
    query,
    flip(next: boolean) {
      query.matches = next
      listener?.()
    },
  }
}

afterEach(() => {
  window.matchMedia = realMatchMedia
})

describe("useIsMobile", () => {
  it("asks for the shell's phone breakpoint and reports a match", () => {
    const media = stubMedia(true)
    expect(renderHook(() => useIsMobile()).result.current).toBe(true)
    expect(media.matchMedia).toHaveBeenCalledWith("(max-width: 767px)")
  })

  it("reports a wide viewport", () => {
    stubMedia(false)
    expect(renderHook(() => useIsMobile()).result.current).toBe(false)
  })

  it("follows the viewport as it crosses the breakpoint and unsubscribes on unmount", () => {
    const media = stubMedia(false)
    const { result, unmount } = renderHook(() => useIsMobile())
    act(() => media.flip(true))
    expect(result.current).toBe(true)
    unmount()
    expect(media.query.removeEventListener).toHaveBeenCalled()
  })

  it("is false where matchMedia does not exist", () => {
    // @ts-expect-error simulating an environment without matchMedia
    window.matchMedia = undefined
    expect(renderHook(() => useIsMobile()).result.current).toBe(false)
  })
})
