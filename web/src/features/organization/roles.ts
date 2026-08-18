/**
 * The fixed roles and statuses the tenancy surface is written in.
 *
 * Fixed roles are the settled OSS line (`MANAGEMENT_ROLES` in
 * `src/gateway/models/tenancy.py`): four roles, of which two manage. Anything
 * finer-grained is overlay depth, so these lists are deliberately literal rather
 * than fetched, and the server refuses a role it does not know either way.
 *
 * The predicates below are the client half of gates the server already enforces.
 * They exist so a control that would always be refused is disabled rather than
 * offered, never to decide who may do what.
 */

import type { OrganizationContext, OrganizationMember } from "@/client"

/** Every role an organization or workspace membership can carry, most privileged first. */
export const MEMBERSHIP_ROLES = ["owner", "admin", "member", "viewer"] as const

/** Every status an organization membership can carry. */
export const MEMBERSHIP_STATUSES = ["active", "invited", "suspended"] as const

/** The roles that may manage an organization or a workspace. */
const MANAGEMENT_ROLES: readonly string[] = ["owner", "admin"]

/** Whether the caller's standing in their organization lets them manage it. */
export function canManage(context: OrganizationContext | undefined): boolean {
  return context !== undefined && MANAGEMENT_ROLES.includes(context.role)
}

/** Whether the caller owns their organization, which is what deleting it takes. */
export function isOwner(context: OrganizationContext | undefined): boolean {
  return context?.role === "owner"
}

/** The active owners of an organization, which is what the last-owner rules count. */
export function activeOwnerCount(
  members: readonly OrganizationMember[],
): number {
  return members.filter(
    (member) => member.role === "owner" && member.status === "active",
  ).length
}

/**
 * Why a membership change would be refused, or undefined if it would be allowed.
 *
 * The two rules are the server's (`_validate_membership_update` in
 * `organization_service.py`): only an owner outranks an owner, and the last
 * active owner cannot be demoted or deactivated, which would leave the
 * organization with nobody able to manage or delete it. Stated here as a
 * sentence so the disabled control says why rather than just refusing.
 */
export function membershipChangeBlockedReason({
  member,
  context,
  members,
}: {
  member: OrganizationMember
  context: OrganizationContext | undefined
  members: readonly OrganizationMember[]
}): string | undefined {
  if (!canManage(context)) {
    return "Only organization owners and admins can change memberships."
  }
  if (!member.organization_member_id) {
    return "This row is a pending invitation, which has no membership to change yet."
  }
  if (member.role === "owner" && !isOwner(context)) {
    return "Only an owner can change another owner's membership."
  }
  if (
    member.role === "owner" &&
    member.status === "active" &&
    activeOwnerCount(members) <= 1
  ) {
    return "This is the last active owner. Promote someone else first."
  }
  return undefined
}

/** How a member is named on screen: their name, then their email, then their id. */
export function memberLabel(member: OrganizationMember): string {
  return (
    member.full_name ??
    member.email ??
    // A standalone operator identity has neither: it is a label, not a sign-in
    // address (M4: "local identities have no email").
    (member.user_id
      ? `Identity ${member.user_id.slice(0, 8)}`
      : "Unknown member")
  )
}
