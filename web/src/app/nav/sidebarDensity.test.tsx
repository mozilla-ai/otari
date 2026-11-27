import { act, renderHook } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import {
  DENSE_SIDEBAR_KEY,
  useDenseSidebarPreference,
} from "@/app/nav/sidebarDensity"

describe("the dense sidebar preference", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("is off until someone turns it on", () => {
    const { result } = renderHook(() => useDenseSidebarPreference())
    expect(result.current[0]).toBe(false)
  })

  it("is kept in localStorage and read back by a later page", () => {
    const first = renderHook(() => useDenseSidebarPreference())
    act(() => first.result.current[1](true))
    expect(window.localStorage.getItem(DENSE_SIDEBAR_KEY)).toBe("1")
    first.unmount()

    const later = renderHook(() => useDenseSidebarPreference())
    expect(later.result.current[0]).toBe(true)
    act(() => later.result.current[1](false))
    expect(window.localStorage.getItem(DENSE_SIDEBAR_KEY)).toBe("0")
  })

  it("moves every reader at once", () => {
    const one = renderHook(() => useDenseSidebarPreference())
    const other = renderHook(() => useDenseSidebarPreference())
    act(() => one.result.current[1](true))
    expect(other.result.current[0]).toBe(true)
    act(() => other.result.current[1](false))
    expect(one.result.current[0]).toBe(false)
  })

  it("follows another tab", () => {
    const { result } = renderHook(() => useDenseSidebarPreference())
    act(() => {
      window.localStorage.setItem(DENSE_SIDEBAR_KEY, "1")
      window.dispatchEvent(
        new StorageEvent("storage", { key: DENSE_SIDEBAR_KEY }),
      )
    })
    expect(result.current[0]).toBe(true)
  })

  it("still works for the page when storage refuses, and is off when it cannot be read", () => {
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("blocked")
    })
    const { result } = renderHook(() => useDenseSidebarPreference())
    act(() => result.current[1](true))
    expect(result.current[0]).toBe(true)
    act(() => result.current[1](false))
    expect(result.current[0]).toBe(false)

    vi.restoreAllMocks()
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("blocked")
    })
    const unreadable = renderHook(() => useDenseSidebarPreference())
    expect(unreadable.result.current[0]).toBe(false)
  })
})
