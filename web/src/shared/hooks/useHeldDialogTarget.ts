import { useEffect, useState } from "react"

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
 *
 * The row is let go `EXIT_MS` after the dialog closes, so a form that holds
 * queries and mutations is unmounted once its exit is over rather than living on
 * with a row that may since have been deleted. The target must be referentially
 * stable between renders (a row found in query data is), because a new object
 * each render reads as a new row.
 */

/** Longer than the dialog's ~100ms exit, with room for a slow frame. */
const EXIT_MS = 400

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

  useEffect(() => {
    if (isOpen) return
    const timer = setTimeout(() => {
      setState((current) =>
        current.isOpen ? current : { ...current, held: undefined },
      )
    }, EXIT_MS)
    return () => clearTimeout(timer)
  }, [isOpen])

  return state
}
