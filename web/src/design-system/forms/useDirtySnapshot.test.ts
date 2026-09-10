import { act, renderHook } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { useDirtySnapshot } from "./useDirtySnapshot"

describe("useDirtySnapshot", () => {
  it("reports a draft that has not moved as clean", () => {
    const { result } = renderHook(() =>
      useDirtySnapshot({ name: "", role: "member" }),
    )

    expect(result.current.isDirty).toBe(false)
  })

  it("sees a field the caller did not have to name", () => {
    // The point of the snapshot: the guard cannot be one edit behind the form,
    // because it holds the whole draft rather than a list of fields.
    const { result, rerender } = renderHook(
      (draft: Record<string, unknown>) => useDirtySnapshot(draft),
      { initialProps: { name: "", role: "member" } },
    )

    rerender({ name: "", role: "viewer" })

    expect(result.current.isDirty).toBe(true)
  })

  it("reports clean again once the draft returns to its seed", () => {
    const { result, rerender } = renderHook(
      (draft: Record<string, unknown>) => useDirtySnapshot(draft),
      { initialProps: { name: "" } },
    )

    rerender({ name: "typed" })
    expect(result.current.isDirty).toBe(true)

    rerender({ name: "" })
    expect(result.current.isDirty).toBe(false)
  })

  it("keeps the seed it was mounted with, so a late default is a change", () => {
    // Deliberate, and the reason a form whose seed resolves after mount has to
    // seed from the resolved value instead: this hook cannot tell an operator's
    // first keystroke from a list arriving.
    const { result, rerender } = renderHook(
      (draft: Record<string, unknown>) => useDirtySnapshot(draft),
      { initialProps: { budgetId: "" } },
    )

    rerender({ budgetId: "budget-1" })

    expect(result.current.isDirty).toBe(true)
  })

  it("re-seeds on reset, for a form that stays open past a save", () => {
    const { result, rerender } = renderHook(
      (draft: Record<string, unknown>) => useDirtySnapshot(draft),
      { initialProps: { name: "" } },
    )

    rerender({ name: "saved" })
    act(() => result.current.reset())
    rerender({ name: "saved" })

    expect(result.current.isDirty).toBe(false)
  })
})
