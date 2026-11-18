import { renderHook } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { useBackForwardCacheRestore } from "@/shared/hooks/useBackForwardCacheRestore"

/** A `pageshow`, which jsdom does not model itself. */
function pageShow(persisted: boolean) {
  const event = new Event("pageshow")
  Object.defineProperty(event, "persisted", { value: persisted })
  window.dispatchEvent(event)
}

describe("useBackForwardCacheRestore", () => {
  it("runs on a restore from the cache", () => {
    const onRestore = vi.fn()
    renderHook(() => useBackForwardCacheRestore(onRestore))

    pageShow(true)

    expect(onRestore).toHaveBeenCalledTimes(1)
  })

  it("does not run on an ordinary load", () => {
    const onRestore = vi.fn()
    renderHook(() => useBackForwardCacheRestore(onRestore))

    pageShow(false)

    expect(onRestore).not.toHaveBeenCalled()
  })

  it("stops listening once unmounted", () => {
    const onRestore = vi.fn()
    const { unmount } = renderHook(() => useBackForwardCacheRestore(onRestore))

    unmount()
    pageShow(true)

    expect(onRestore).not.toHaveBeenCalled()
  })
})
