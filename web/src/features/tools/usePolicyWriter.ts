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
  const [seenKey, setSeenKey] = useState(resetKey)

  if (resetKey !== seenKey) {
    setSeenKey(resetKey)
    carried.current = undefined
  }

  const commit = (patch: Partial<Body>) => {
    const base = carried.current ?? server
    if (base === undefined) {
      return Promise.reject(
        new Error("The policy has not been read yet, so nothing can be saved."),
      )
    }
    const run = queue.current.then(async () => {
      // Read inside the chained callback, not outside it: a commit queued
      // behind another must build on what that one stored, not on the base
      // that existed when it was queued.
      const from = carried.current ?? base
      carried.current = await put({ ...toBody(from), ...patch })
    })
    queue.current = run.catch(() => undefined)
    return run
  }

  return commit
}
