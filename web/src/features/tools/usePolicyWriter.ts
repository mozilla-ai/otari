import { useRef, useState } from "react"

/**
 * Serialized writes for a policy that is stored as one row.
 *
 * Both workspace groups PUT the whole policy, so a row's save has to carry the
 * other rows with it. Autosave is what makes that dangerous: each control has
 * its own save state by design, so two rows can be in flight at once, and a
 * body built from the last value the *query* returned would still hold the
 * pre-first-write state. Leaving one field and immediately editing another
 * reverted the first.
 *
 * So the base for each body is the previous write's own response rather than
 * the query, and commits are chained rather than fired in parallel. A failed
 * write does not poison the chain: the next commit still runs, from the last
 * base known good.
 */
export function usePolicyWriter<Body, Stored>({
  server,
  resetKey,
  toBody,
  put,
}: {
  /** The policy as the query last returned it. */
  server: Stored | undefined
  /** Changing it drops the carried base, for when the row is a different one. */
  resetKey: string
  /**
   * The writable half of the stored row. Explicit rather than a spread: a
   * stored policy also carries `workspace_id`, `configured`, the server's own
   * `allowed_images` and `available_tools` and two timestamps, none of which
   * belongs in a PUT body.
   */
  toBody: (stored: Stored) => Body
  put: (body: Body) => Promise<Stored>
}) {
  const carried = useRef<Stored | undefined>(undefined)
  const queue = useRef<Promise<unknown>>(Promise.resolve())
  // Which row the carried base belongs to. A write that was already in flight
  // when the row changed still resolves, and without this it would store its
  // answer as the new row's base: the next commit would build a body out of the
  // old row's values and PUT it to the new one.
  const era = useRef(0)
  const [seenKey, setSeenKey] = useState(resetKey)

  if (resetKey !== seenKey) {
    setSeenKey(resetKey)
    carried.current = undefined
    era.current += 1
    // A commit for the new row must not queue behind the old row's write. The
    // cost is that the queue stops being a single serialization point across a
    // reset: a callback already pending on the old promise still runs, beside
    // the new row's chain. The `startedIn` bail-out in `commit` is what carries
    // that weight, which is why it guards the read and the send, not just the
    // store.
    queue.current = Promise.resolve()
  }

  const commit = (patch: Partial<Body>) => {
    const base = carried.current ?? server
    if (base === undefined) {
      return Promise.reject(
        new Error("The policy has not been read yet, so nothing can be saved."),
      )
    }
    const startedIn = era.current
    const run = queue.current.then(async () => {
      // Superseded before it got its turn, so it neither reads the shared base
      // nor sends. Guarding only the store would be too late: this callback is
      // orphaned on the previous queue and runs alongside the new row's chain,
      // so by now `carried.current` can already hold the new row's values, and
      // building a body out of them would PUT the new row's policy through the
      // old row's `put`. Resolves rather than rejects: the operator navigated
      // away, which is not a failed save.
      if (startedIn !== era.current) return
      // Read inside the chained callback, not outside it: a commit queued
      // behind another must build on what that one stored, not on the base
      // that existed when it was queued.
      const from = carried.current ?? base
      const stored = await put({ ...toBody(from), ...patch })
      // Checked again on the way out: the row can change while this is in
      // flight, and the answer belongs to a row nothing is looking at.
      if (startedIn === era.current) carried.current = stored
    })
    queue.current = run.catch(() => undefined)
    return run
  }

  return commit
}
