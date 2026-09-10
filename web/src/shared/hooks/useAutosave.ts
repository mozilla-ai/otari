import { useState } from "react"

import { errorMessage } from "@/design-system/feedback/errorMessage"

/**
 * Enter commits by leaving the field, so a keyboard save and a pointer save run
 * the same blur path rather than two that can disagree about what changed.
 */
export function commitOnEnter(event: React.KeyboardEvent<HTMLInputElement>) {
  if (event.key === "Enter") {
    event.preventDefault()
    event.currentTarget.blur()
  }
}

export interface Autosave {
  /** In flight, so the control disables itself rather than taking a second edit. */
  isSaving: boolean
  error: string
  /** Commits a value and drives the row's state. Never rejects. */
  run: (commit: () => Promise<unknown>) => Promise<void>
}

/**
 * The save state of one autosaving control: in flight, or refused.
 *
 * Per control rather than per page, because a mutation hook's `isPending` is
 * shared by every row that uses it and would disable all of them.
 *
 * A save that worked says nothing: a confirmation on every row of a page where
 * everything saves itself is a mark the reader learns to ignore. A refusal is
 * the outcome worth a word, and it stays until the next attempt, since the
 * value that caused it is still in the field.
 */
export function useAutosave(): Autosave {
  const [isSaving, setIsSaving] = useState(false)
  const [error, setError] = useState("")

  const run = async (commit: () => Promise<unknown>) => {
    setError("")
    setIsSaving(true)
    try {
      await commit()
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setIsSaving(false)
    }
  }

  return { isSaving, error, run }
}
