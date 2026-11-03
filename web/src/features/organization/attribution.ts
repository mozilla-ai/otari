import { useMemo } from "react"

import type { OrganizationMember } from "@/client"
import {
  useOrganizationContext,
  useOrganizationMembers,
} from "@/shared/api/organizations"

import { memberLabel } from "./roles"

// Members and spend identities are two tables that have not merged yet. Keys,
// budgets, and usage attach to the gateway's string-keyed `users` row; a member
// is a UUID identity. `attribution_user_id` is the join between them, minted
// when the member is created, and it is what lets a key be issued to a person
// rather than to an opaque id. otari-ai#1727 decides how the two converge, and
// this map becomes the identity function and can go when they do.
//
// A member whose `attribution_user_id` is null has no usable row (nobody minted
// one, or it was soft-deleted through `DELETE /v1/users`), and key
// creation would refuse them. Those are left out rather than mapped, so nothing
// offers an owner the server will reject.
export function memberLabelsByAttributionId(
  members: OrganizationMember[] | undefined,
): ReadonlyMap<string, string> {
  return new Map(
    (members ?? []).flatMap((member): [string, string][] =>
      member.attribution_user_id
        ? [[member.attribution_user_id, memberLabel(member)]]
        : [],
    ),
  )
}

/**
 * Names the people behind the request-plane owner ids that keys and usage carry.
 */
export function useMemberAttributionLabels(): ReadonlyMap<string, string> {
  const members = useOrganizationMembers()
  return useMemo(
    () => memberLabelsByAttributionId(members.data),
    [members.data],
  )
}

/** Sign-in addresses by the owner id they bill through, for members who have one. */
function memberEmailsByAttributionId(
  members: OrganizationMember[] | undefined,
): ReadonlyMap<string, string> {
  return new Map(
    (members ?? []).flatMap((member): [string, string][] =>
      member.attribution_user_id && member.email
        ? [[member.attribution_user_id, member.email]]
        : [],
    ),
  )
}

/** The addresses behind `useMemberAttributionLabels`, from the same roster read. */
export function useMemberAttributionEmails(): ReadonlyMap<string, string> {
  const members = useOrganizationMembers()
  return useMemo(
    () => memberEmailsByAttributionId(members.data),
    [members.data],
  )
}

/**
 * The owner id the signed-in person bills through, with whether it is still
 * being resolved. `ownerId` is "" while pending, for a member with no
 * attribution row, and when the roster could not be read (`isFailed`), so a
 * caller that must tell those apart reads the flags.
 *
 * A failed roster read is settled from its first failure rather than after the
 * query's retries: a caller waiting on the owner id would otherwise hold its
 * page for those seconds, and it already has an answer, which is "not known".
 */
export function useCurrentOwner(enabled = true): {
  ownerId: string
  isPending: boolean
  isFailed: boolean
} {
  const context = useOrganizationContext()
  // The roster is asked once the context has answered: a caller it failed for
  // has no organization to read it from.
  const rosterEnabled = enabled && context.data !== undefined
  const members = useOrganizationMembers(rosterEnabled)
  const callerId = context.data?.caller?.user_id
  const memberId = context.data?.organization_member_id
  const ownerId = useMemo(() => {
    const me = (members.data ?? []).find(
      (member) =>
        (callerId && member.user_id === callerId) ||
        (memberId && member.organization_member_id === memberId),
    )
    return me?.attribution_user_id ?? ""
  }, [members.data, callerId, memberId])
  const isFailed =
    rosterEnabled &&
    members.data === undefined &&
    (members.isError || members.failureCount > 0)
  return {
    ownerId,
    isPending:
      enabled &&
      (context.isPending || (rosterEnabled && members.isPending && !isFailed)),
    isFailed,
  }
}

/**
 * The owner id the signed-in person bills through. Undefined while the reads
 * are pending or failed, and for a member with no attribution row.
 */
export function useCurrentOwnerId(): string | undefined {
  return useCurrentOwner().ownerId || undefined
}
