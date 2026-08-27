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
 *
 * Two of them are deliberately **narrower** than the server, and both narrow in
 * the safe direction (a control is disabled where the server would have
 * allowed it, never the reverse). Neither is reachable while a standalone
 * deployment has one operator identity, who is a superuser and an owner of the
 * organization and of every workspace in it; both become reachable with the
 * per-user sign-in of otari-ai#1716, which is when to revisit them.
 *
 * - **`canManage` alone is organization-role-only.** `_require_workspace_
 *   management_access` also grants a caller whose *workspace* membership is an
 *   active owner or admin, so an organization member who owns one workspace may
 *   manage it. `canManage` itself stays narrower on the organization pages,
 *   which have no selected workspace to ask about: answering there would need
 *   the caller's own user id resolved through the roster, which is a second
 *   query to say what the server will say anyway. `canManageWorkspace` below is
 *   for the workspace-scoped pages, where `CallerWorkspaceMembershipPublic.role`
 *   (the switcher's own data, already fetched) already names the caller's
 *   standing in *that* workspace, with nothing more to resolve.
 * - **Superuser is not on the contract.** The server grants organization
 *   management, and deletion, to `user.is_superuser` whatever their role;
 *   `OrganizationMembershipContextPublic` carries no such field, so nothing here
 *   can see it. otari.ai's own `useIsOrgAdmin` does check it, which is the
 *   divergence to close when the context grows the field.
 */

import type {
  MembershipRole,
  OrganizationContext,
  OrganizationMember,
  SettableMemberStatus,
} from "@/client"

/** Every role an organization or workspace membership can carry, most privileged first. */
export const MEMBERSHIP_ROLES: readonly MembershipRole[] = [
  "owner",
  "admin",
  "member",
  "viewer",
]

/**
 * The statuses a membership may be *given*, which is narrower than the set it
 * may *hold*.
 *
 * Not offered as a picker anywhere, and the roster page says why: suspending is
 * what Remove does, and a suspended membership leaves the roster, so neither
 * direction has a control. Kept because it is the shape of a rule worth not
 * rediscovering: "invited" is a stored status the invitation flow produces, the
 * gateway refuses it on the update request
 * (`OrganizationMemberSettableStatus`), and whatever control arrives with that
 * flow has to respect the same line.
 */
export const SETTABLE_MEMBER_STATUSES: readonly SettableMemberStatus[] = [
  "active",
  "suspended",
]

/**
 * Narrow a picker's string back to the vocabulary the gateway publishes.
 *
 * A `<select>` hands back a bare string, and the request types are unions now,
 * so something has to bridge the two. A guard rather than a cast: the options
 * come from the constants above today, and a value that is not in them is a bug
 * worth dropping the write for rather than sending and having refused.
 */
export function asMembershipRole(value: string): MembershipRole | undefined {
  return MEMBERSHIP_ROLES.find((role) => role === value)
}

export function asSettableStatus(
  value: string,
): SettableMemberStatus | undefined {
  return SETTABLE_MEMBER_STATUSES.find((status) => status === value)
}

/** A vocabulary value as it is shown: "owner" is a wire value, "Owner" is a label. */
export function membershipLabel(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1)
}

/** The roles that may manage an organization or a workspace. */
const MANAGEMENT_ROLES: readonly string[] = ["owner", "admin"]

/** Whether the caller's standing in their organization lets them manage it. */
export function canManage(context: OrganizationContext | undefined): boolean {
  return context !== undefined && MANAGEMENT_ROLES.includes(context.role)
}

/**
 * Whether this caller also operates the deployment, which no role above confers.
 *
 * The other authority a signed-in identity can hold, and the reason the roster's
 * role picker looked like it did more than it does (otari#838): an organization
 * role is authority over one tenant, and the deployment's own surfaces answer to
 * `is_superuser` or the bootstrap identity instead.
 *
 * Like the predicates above, this is the client half of a gate the server
 * enforces, and it exists so a control that would be refused is withheld rather
 * than offered. It reads the same field `GET /v1/admin/access` publishes, which
 * `OrganizationMembershipContextPublic` now carries so a page that already has
 * the context needs no second request to know.
 *
 * Requires an explicit `true`: an absent or still-loading context is not an
 * operator, which withholds a deployment-wide read for one paint rather than
 * firing it and rendering its refusal.
 */
export function isDeploymentOperator(
  context: OrganizationContext | undefined,
): boolean {
  return context?.deployment_operator === true
}

/**
 * Whether the caller may manage a specific workspace: `canManage`'s
 * organization arm, or an active owner/admin of that workspace itself.
 *
 * `workspaceRole` is the caller's own role in the selected workspace (e.g.
 * `useSelectedWorkspace().selected?.role`), not a roster entry. See the
 * module docstring for why that is enough on its own, with no extra request.
 */
export function canManageWorkspace(
  context: OrganizationContext | undefined,
  workspaceRole: string | undefined,
): boolean {
  return (
    canManage(context) ||
    (workspaceRole !== undefined && MANAGEMENT_ROLES.includes(workspaceRole))
  )
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
