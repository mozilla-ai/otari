import { useCallback, useMemo } from "react"

import { useCurrentOwner } from "@/features/organization/attribution"
import { useUrlState } from "@/shared/helpers/urlState"

export type ObserveScope = "mine" | "everyone"

const URL_DEFAULTS = { scope: "", user_id: "", page: "0" } as const

/**
 * Whose requests the Overview shows. Everyone's, narrowed by the User filter,
 * unless the URL carries `scope=mine`, which reads the caller's own usage
 * identity. A caller without one falls back to Everyone. Until that identity
 * has resolved, `isResolved` is false and a page must not fetch, or Mine would
 * flash team data. `active: false` is for a page outside the Overview, which
 * reads the URL's `user_id`.
 */
export function useObserveScope(active = true) {
  const url = useUrlState(URL_DEFAULTS)
  const wantsMine = active && url.get("scope") === "mine"
  const owner = useCurrentOwner(wantsMine)

  const urlUserIds = url.getAll("user_id")
  const hasOwner = owner.ownerId !== ""
  const scope: ObserveScope =
    wantsMine && (owner.isPending || hasOwner) ? "mine" : "everyone"
  const isResolved = !wantsMine || !owner.isPending

  const userIds = useMemo(
    () => (scope === "mine" && hasOwner ? [owner.ownerId] : urlUserIds),
    [scope, hasOwner, owner.ownerId, urlUserIds],
  )

  // What a link out of the page carries so the destination opens on the same
  // scope.
  const scopeSearch = useMemo(
    () => (scope === "mine" ? { scope } : {}),
    [scope],
  )

  const patch = url.patch
  // Mine hides the URL's users, so they go with it rather than resurfacing.
  const clearMine = useCallback(
    () => patch({ scope: "", user_id: [], page: 0 }),
    [patch],
  )

  return {
    scope,
    /** Drops `scope=mine` and the users it hid, back to Everyone. */
    clearMine,
    ownerId: owner.ownerId,
    isResolved,
    /** The `user_id` filter to fetch with, whatever the URL carries. */
    userIds,
    scopeSearch,
  }
}
