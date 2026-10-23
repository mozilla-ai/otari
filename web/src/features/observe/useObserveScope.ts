import { useCallback, useMemo } from "react"

import { useCurrentOwner } from "@/features/organization/attribution"
import { canManage } from "@/features/organization/roles"
import {
  useDeploymentOperator,
  useOrganizationContext,
} from "@/shared/api/organizations"
import { useUrlState } from "@/shared/helpers/urlState"

export type ObserveScope = "mine" | "everyone"

const URL_DEFAULTS = { scope: "", user_id: "", page: "0" } as const

const isScope = (value: string): value is ObserveScope =>
  value === "mine" || value === "everyone"

/**
 * Whose requests the Overview shows: the caller's own usage identity
 * ("mine") or everything their role may read ("everyone").
 *
 * The URL key `scope` wins; absent, a caller who neither operates the
 * deployment nor manages the organization opens on Mine and everyone else on
 * Everyone. Mine needs the caller's owner id, so a caller without one falls
 * back to Everyone, as does one whose roster read failed. Until the role (and,
 * for Mine, the owner id) has resolved, `isResolved` is false and a page must
 * not fetch, or a member would see a flash of team data. `active: false` is
 * for a page outside the Overview, which reads the URL's `user_id` as it
 * always did.
 */
export function useObserveScope(active = true) {
  const url = useUrlState(URL_DEFAULTS)
  const operator = useDeploymentOperator()
  const context = useOrganizationContext()
  const owner = useCurrentOwner(active)

  const requested = url.get("scope")
  const urlUserIds = url.getAll("user_id")
  const hasOwner = owner.ownerId !== ""
  const defaultsToMine =
    operator.isSettled &&
    !operator.isOperator &&
    operator.answer !== "no-organization" &&
    !canManage(context.data)

  const wantsMine = isScope(requested) ? requested === "mine" : defaultsToMine
  const scope: ObserveScope =
    active && wantsMine && (owner.isPending || hasOwner) ? "mine" : "everyone"

  const isResolved =
    !active ||
    requested === "everyone" ||
    ((isScope(requested) || operator.isSettled) &&
      (!wantsMine || !owner.isPending))

  const userIds = useMemo(
    () => (scope === "mine" && hasOwner ? [owner.ownerId] : urlUserIds),
    [scope, hasOwner, owner.ownerId, urlUserIds],
  )

  // What a link out of the page carries so the destination opens on the same
  // scope: an explicit choice, or Mine where it is only the role's default.
  const carried = isScope(requested)
    ? requested
    : scope === "mine"
      ? "mine"
      : ""
  const scopeSearch = useMemo(
    () => (carried ? { scope: carried } : {}),
    [carried],
  )

  const patch = url.patch
  const setScope = useCallback(
    (next: ObserveScope) =>
      patch({
        scope: next,
        ...(next === "mine" ? { user_id: [] } : {}),
        page: 0,
      }),
    [patch],
  )

  return {
    scope,
    setScope,
    ownerId: owner.ownerId,
    hasOwner,
    /** Mine cannot be offered: the owner id read has settled without one. */
    isMineUnavailable: !hasOwner && !owner.isPending,
    /** The roster could not be read, so whether the caller has an owner id is unknown. */
    isOwnerUnknown: owner.isFailed,
    isResolved,
    /** The `user_id` filter to fetch with, whatever the URL carries. */
    userIds,
    scopeSearch,
  }
}
