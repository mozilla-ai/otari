import { useState } from "react"

/**
 * The row an edit dialog is about, kept through the dialog's exit.
 *
 * An edit form mounted as `{row ? <Form isOpen /> : null}` is gone the moment
 * `row` clears, so it never plays its exit. This hands back the last row for as
 * long as the dialog is on screen, so the page can keep the form mounted and
 * drive it with `isOpen` instead.
 *
 * `generation` counts openings. A draft is fresh on every open
 * (`design/feedback.md`), and a key on the row alone would hand the next Edit of
 * the same row the draft the last one left behind, so the form is keyed on both.
 * It does not move while the row's object is replaced by a refetch, which would
 * remount the form under the operator.
 */
export function useHeldDialogTarget<T>(target: T | undefined): {
  held: T | undefined
  isOpen: boolean
  generation: number
} {
  const isOpen = target !== undefined
  const [state, setState] = useState({
    held: target,
    isOpen,
    generation: isOpen ? 1 : 0,
  })

  // Derived state, adjusted while rendering rather than in an effect, so the
  // frame that opens the dialog already has the row.
  if (isOpen !== state.isOpen || (isOpen && target !== state.held)) {
    setState({
      held: isOpen ? target : state.held,
      isOpen,
      generation:
        isOpen && !state.isOpen ? state.generation + 1 : state.generation,
    })
  }

  return state
}
