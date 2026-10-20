import { act, renderHook } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { useClientPagination } from "./useClientPagination"

const rows = (count: number) => Array.from({ length: count }, (_, i) => i)

describe("useClientPagination", () => {
  it("pages 32 rows at 15 into three pages, the last holding two", () => {
    const { result } = renderHook(() => useClientPagination(rows(32)))
    expect(result.current.pageRows).toEqual(rows(15))
    expect(result.current.pagination.total).toBe(32)
    act(() => result.current.pagination.onPageChange(2))
    expect(result.current.pageRows).toEqual([30, 31])
    expect(result.current.pagination.rowsOnPage).toBe(2)
  })

  it("goes back to the first page when the page size changes", () => {
    const { result } = renderHook(() => useClientPagination(rows(32)))
    act(() => result.current.pagination.onPageChange(1))
    act(() => result.current.pagination.onPageSizeChange(30))
    expect(result.current.pagination.page).toBe(0)
    expect(result.current.pageRows).toHaveLength(30)
  })

  it("shows the last page when the rows shrink under the one it was on", () => {
    const { result, rerender } = renderHook(
      ({ count }) => useClientPagination(rows(count)),
      { initialProps: { count: 32 } },
    )
    act(() => result.current.pagination.onPageChange(2))
    rerender({ count: 20 })
    expect(result.current.pagination.page).toBe(1)
    expect(result.current.pageRows).toEqual([15, 16, 17, 18, 19])
    // Growing back does not return to the page that was past the end.
    rerender({ count: 32 })
    expect(result.current.pagination.page).toBe(1)
  })

  it("goes back to the first page when the reset key changes, not the rows", () => {
    const { result, rerender } = renderHook(
      ({ resetKey }) => useClientPagination(rows(32), resetKey),
      { initialProps: { resetKey: "a" } },
    )
    act(() => result.current.pagination.onPageChange(1))
    rerender({ resetKey: "a" })
    expect(result.current.pagination.page).toBe(1)
    rerender({ resetKey: "b" })
    expect(result.current.pagination.page).toBe(0)
  })

  it("lands a reset on the first page even when the rows shrink with it", () => {
    // Health: 32 results on page 3, then a filter down to 20 failing ones.
    const { result, rerender } = renderHook(
      ({ count, resetKey }) => useClientPagination(rows(count), resetKey),
      { initialProps: { count: 32, resetKey: "all" } },
    )
    act(() => result.current.pagination.onPageChange(2))
    rerender({ count: 20, resetKey: "failing" })
    expect(result.current.pagination.page).toBe(0)
    rerender({ count: 20, resetKey: "failing" })
    expect(result.current.pagination.page).toBe(0)
  })
})
