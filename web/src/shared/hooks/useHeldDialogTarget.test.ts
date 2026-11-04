import { renderHook } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { useHeldDialogTarget } from "@/shared/hooks/useHeldDialogTarget"

type Row = { id: string }

describe("useHeldDialogTarget", () => {
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
})
