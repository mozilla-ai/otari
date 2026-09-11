import { useRef } from "react"

/**
 * Whether a form's draft still matches what it was seeded with.
 *
 * One snapshot of the whole draft, rather than a predicate naming each field:
 * a hand-listed predicate is one edit behind the form the moment a field is
 * added, and every guard in this dashboard that silently discarded a choice
 * had drifted that way (a default member budget, a role, a provider, a pasted
 * key). Pass everything the operator can change; the shape is the guard.
 *
 * The seed is the draft on mount, so the component that owns the draft has to
 * be the one that remounts per open. See feedback.md, "A draft is fresh on
 * every open and untouched through the exit".
 *
 * `reset` re-seeds to the draft as it stands, for a form that stays open past
 * a save and should read clean again afterwards.
 */
export function useDirtySnapshot(draft: unknown): {
  isDirty: boolean
  reset: () => void
} {
  const snapshot = JSON.stringify(draft)
  const seeded = useRef(snapshot)
  return {
    isDirty: snapshot !== seeded.current,
    reset: () => {
      seeded.current = snapshot
    },
  }
}
