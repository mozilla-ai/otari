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

  it("keeps one `reset`, so an effect can depend on it", () => {
    // A `reset` that changed identity every render would re-seed on every
    // render through an effect that depends on it, and the form would never
    // read dirty.
    const { result, rerender } = renderHook(
      (draft: Record<string, unknown>) => useDirtySnapshot(draft),
      { initialProps: { name: "" } },
    )
    const first = result.current.reset

    rerender({ name: "typed" })

    expect(result.current.reset).toBe(first)
  })

  it("re-seeds to the latest draft, not the one at the reset's own render", () => {
    // The late-default case: the seed a form is compared against has to be the
    // value that landed, which is why `reset` reads a ref rather than closing
    // over a render.
    const { result, rerender } = renderHook(
      (draft: Record<string, unknown>) => useDirtySnapshot(draft),
      { initialProps: { workspaceIds: [] as string[] } },
    )
    const reset = result.current.reset

    rerender({ workspaceIds: ["ws-1"] })
    act(() => reset())
    rerender({ workspaceIds: ["ws-1"] })

    expect(result.current.isDirty).toBe(false)
  })

  it("seeds an explicit draft, for a default that lands after mount", () => {
    // The caller holds the value the render it is in does not: seeding it there
    // is what keeps a roster answering from reading as the operator's first
    // change. An effect cannot do this: a ref written after the commit re-seeds
    // nothing already rendered.
    const { result, rerender } = renderHook(
      (draft: Record<string, unknown>) => useDirtySnapshot(draft),
      { initialProps: { workspaceIds: [] as string[] } },
    )

    act(() => result.current.reset({ workspaceIds: ["ws-1"] }))
    rerender({ workspaceIds: ["ws-1"] })

    expect(result.current.isDirty).toBe(false)
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
