import { useCallback, useRef, useState } from "react"

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
 * holds the old one. Call it there, during the render that applies the default:
 * the seed is state, so a reset during render re-runs the component and the
 * `isDirty` the caller receives is the corrected one. From an effect it also
 * works, one render later.
 *
 * A reset during render has to be guarded by whatever made the default arrive
 * (`if (!seeded && rows.length > 0)`), the same as any other set-state-while-
 * rendering: unconditional, it is an endless render.
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
  // State rather than a ref, so a reset during render corrects that render.
  // Held in a ref it corrected nothing already rendered, and the two callers
  // that appeared to work did so only because they set other state in the same
  // block, which is what re-ran the component.
  const [seed, setSeed] = useState(snapshot)
  // Read by `reset`, which has no dependencies and so cannot close over the
  // current render's snapshot.
  const latest = useRef(snapshot)
  latest.current = snapshot
  // Not reflexive memoization, which the frontend standards rule out: the
  // compiler memoizes nothing in this hook (it bails with "Cannot access refs
  // during render" on the `latest` read above, measured with
  // babel-plugin-react-compiler 1.0.0 against both this version and the one
  // before it), so this `useCallback` is what actually keeps `reset` stable.
  const reset = useCallback((next?: unknown) => {
    setSeed(next === undefined ? latest.current : JSON.stringify(next))
  }, [])
  return { isDirty: snapshot !== seed, reset }
}
