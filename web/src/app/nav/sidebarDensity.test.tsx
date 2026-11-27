import { act, renderHook } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import {
  DENSE_SIDEBAR_KEY,
  useDenseSidebarPreference,
} from "@/app/nav/sidebarDensity"

/** What another tab does to storage, which is also how a held value is let go. */
function storageChangedElsewhere(key: string | null = DENSE_SIDEBAR_KEY) {
  act(() => {
    window.dispatchEvent(new StorageEvent("storage", { key }))
  })
}

describe("the dense sidebar preference", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
    // The module remembers a value a refused write could not store; any
    // storage event is what lets it go, so each test starts from storage.
    storageChangedElsewhere(null)
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

  it("follows another tab turning it on", () => {
    const { result } = renderHook(() => useDenseSidebarPreference())
    window.localStorage.setItem(DENSE_SIDEBAR_KEY, "1")
    storageChangedElsewhere()
    expect(result.current[0]).toBe(true)
  })

  it("follows another tab clearing storage, which names no key", () => {
    window.localStorage.setItem(DENSE_SIDEBAR_KEY, "1")
    const { result } = renderHook(() => useDenseSidebarPreference())
    expect(result.current[0]).toBe(true)

    window.localStorage.clear()
    storageChangedElsewhere(null)
    expect(result.current[0]).toBe(false)
  })

  it("ignores a storage event about some other key", () => {
    const { result } = renderHook(() => useDenseSidebarPreference())
    act(() => result.current[1](true))
    window.localStorage.removeItem(DENSE_SIDEBAR_KEY)
    storageChangedElsewhere("otari.dashboard.sidebarCollapsed")
    expect(result.current[0]).toBe(true)
  })

  it("still works for the page when storage refuses the write", () => {
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("blocked")
    })
    const { result } = renderHook(() => useDenseSidebarPreference())
    act(() => result.current[1](true))
    expect(result.current[0]).toBe(true)
    act(() => result.current[1](false))
    expect(result.current[0]).toBe(false)
  })

  it("stops holding a refused value once another tab writes", () => {
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("blocked")
    })
    const { result } = renderHook(() => useDenseSidebarPreference())
    act(() => result.current[1](true))
    expect(result.current[0]).toBe(true)

    vi.restoreAllMocks()
    window.localStorage.setItem(DENSE_SIDEBAR_KEY, "0")
    storageChangedElsewhere()
    expect(result.current[0]).toBe(false)
  })

  it("is off when storage cannot be read", () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("blocked")
    })
    window.localStorage.setItem(DENSE_SIDEBAR_KEY, "1")
    const { result } = renderHook(() => useDenseSidebarPreference())
    expect(result.current[0]).toBe(false)
  })
})
