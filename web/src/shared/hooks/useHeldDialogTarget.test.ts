import { act, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { useHeldDialogTarget } from "@/shared/hooks/useHeldDialogTarget"

type Row = { id: string }

describe("useHeldDialogTarget", () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it("keeps the last row once the target clears, so the exit has a body", () => {
    const row: Row = { id: "a" }
    const { result, rerender } = renderHook(
      ({ target }: { target: Row | undefined }) => useHeldDialogTarget(target),
      { initialProps: { target: row as Row | undefined } },
    )
    expect(result.current).toMatchObject({ held: row, isOpen: true })

    rerender({ target: undefined })
    expect(result.current).toMatchObject({ held: row, isOpen: false })
  })

  it("counts every opening, including the same row twice", () => {
    const row: Row = { id: "a" }
    const { result, rerender } = renderHook(
      ({ target }: { target: Row | undefined }) => useHeldDialogTarget(target),
      { initialProps: { target: undefined as Row | undefined } },
    )
    expect(result.current.generation).toBe(0)

    rerender({ target: row })
    expect(result.current.generation).toBe(1)
    rerender({ target: undefined })
    expect(result.current.generation).toBe(1)
    rerender({ target: row })
    expect(result.current.generation).toBe(2)
  })

  it("does not count a refetched copy of the open row as a new opening", () => {
    const { result, rerender } = renderHook(
      ({ target }: { target: Row | undefined }) => useHeldDialogTarget(target),
      { initialProps: { target: { id: "a" } as Row | undefined } },
    )
    rerender({ target: { id: "a" } })
    expect(result.current.generation).toBe(1)
  })

  it("lets the row go once the exit is over, and not before", () => {
    const row: Row = { id: "a" }
    const { result, rerender } = renderHook(
      ({ target }: { target: Row | undefined }) => useHeldDialogTarget(target),
      { initialProps: { target: row as Row | undefined } },
    )
    rerender({ target: undefined })

    act(() => {
      vi.advanceTimersByTime(100)
    })
    expect(result.current.held).toBe(row)

    act(() => {
      vi.advanceTimersByTime(400)
    })
    expect(result.current.held).toBeUndefined()
  })

  it("keeps the row if the dialog is reopened during the exit", () => {
    const row: Row = { id: "a" }
    const { result, rerender } = renderHook(
      ({ target }: { target: Row | undefined }) => useHeldDialogTarget(target),
      { initialProps: { target: row as Row | undefined } },
    )
    rerender({ target: undefined })
    rerender({ target: row })

    act(() => {
      vi.advanceTimersByTime(1000)
    })
    expect(result.current).toMatchObject({ held: row, isOpen: true })
  })

  it("follows a switch from one row to another without a new opening", () => {
    const a: Row = { id: "a" }
    const b: Row = { id: "b" }
    const { result, rerender } = renderHook(
      ({ target }: { target: Row | undefined }) => useHeldDialogTarget(target),
      { initialProps: { target: a as Row | undefined } },
    )
    rerender({ target: b })
    expect(result.current).toMatchObject({ held: b, generation: 1 })
  })
})
