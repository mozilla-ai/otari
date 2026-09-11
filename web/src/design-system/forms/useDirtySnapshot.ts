import { useCallback, useRef } from "react"

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
 * `reset()` re-seeds to the draft as it stands, for a form that stays open past
 * a save and should read clean again afterwards. `reset(next)` seeds an
 * explicit draft instead, which is what a form whose own default lands after
 * mount needs: that default is part of the seed rather than a change, and the
 * code that computes it holds the new value while the render it is in still
 * holds the old one. Seed it there, during the render that applies it, and not
 * from an effect: a ref write after the commit re-seeds nothing that has
 * already been rendered, so the guard stays armed until something else
 * re-renders.
 *
 * `reset`'s identity is stable, so an effect may depend on it; a `reset` that
 * changed every render would re-seed on every render and the form would never
 * read dirty at all.
 */
export function useDirtySnapshot(draft: unknown): {
  isDirty: boolean
  reset: (next?: unknown) => void
} {
  const snapshot = JSON.stringify(draft)
  const seeded = useRef(snapshot)
  // Read by `reset`, which has no dependencies and so cannot close over the
  // current render's snapshot.
  const latest = useRef(snapshot)
  latest.current = snapshot
  const reset = useCallback((next?: unknown) => {
    seeded.current = next === undefined ? latest.current : JSON.stringify(next)
  }, [])
  return { isDirty: snapshot !== seeded.current, reset }
}
