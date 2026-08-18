import { useMemo } from "react"

import type { OrganizationMember } from "@/client"
import { useOrganizationMembers } from "@/shared/api/hooks"
import { useSurfaces } from "@/shared/hooks/useDeployment"

import { memberLabel } from "./roles"

// Members and users are two tables that have not merged yet. Keys, budgets, and
// usage attach to the gateway's string-keyed `users` row; a member is a UUID
// identity. `attribution_user_id` is the join between them, minted when the
// member is created, and it is what lets a key be issued to a person rather than
// to an opaque id. M4 collapses the two, at which point this map becomes the
// identity function and can go.
//
// A member whose `attribution_user_id` is null has no usable row (nobody minted
// one, or it was soft-deleted through the users page), and key creation would
// refuse them. Those are left out rather than mapped, so nothing offers an owner
// the server will reject.
export function memberLabelsByAttributionId(
  members: OrganizationMember[] | undefined,
): ReadonlyMap<string, string> {
  const labels = new Map<string, string>()
  for (const member of members ?? []) {
    if (member.attribution_user_id) {
      labels.set(member.attribution_user_id, memberLabel(member))
    }
  }
  return labels
}

/**
 * Names the people behind the request-plane owner ids that keys and usage carry.
 *
 * Gated on the `organizations` surface: a deployment that does not host the
 * roster has no route to ask, and an owner id then stays the bare string it
 * always was, which is what every page rendered before members existed.
 */
export function useMemberAttributionLabels(): ReadonlyMap<string, string> {
  const hasSurface = useSurfaces()
  const members = useOrganizationMembers(hasSurface("organizations"))
  return useMemo(
    () => memberLabelsByAttributionId(members.data),
    [members.data],
  )
}
