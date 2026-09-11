import { act, render, renderHook } from "@testing-library/react"
import { useRef } from "react"
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

  it("corrects the same render a reset(next) was called in", () => {
    // The precondition that used to be hidden: held in a ref, the seed fixed
    // nothing already rendered, and a caller that reseeded during render stayed
    // armed unless it happened to set other state in the same block. The seed
    // is state now, so this render is re-run and the `isDirty` the caller
    // receives is the corrected one, with no other update in the block.
    const seen: boolean[] = []
    function Host({ landed }: { landed: string[] }) {
      const { isDirty, reset } = useDirtySnapshot({ workspaceIds: landed })
      const first = useRef(true)
      if (first.current && landed.length > 0) {
        first.current = false
        reset({ workspaceIds: landed })
      }
      seen.push(isDirty)
      return null
    }

    const { rerender } = render(<Host landed={[]} />)
    rerender(<Host landed={["ws-1"]} />)

    // The last render the caller was handed reads clean.
    expect(seen.at(-1)).toBe(false)
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
