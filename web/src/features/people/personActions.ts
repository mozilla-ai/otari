import type { OrganizationContext, OrganizationMember } from "@/client"
import {
  canManage,
  membershipChangeBlockedReason,
} from "@/features/organization/roles"

interface PersonAction {
  /** Why it would be refused, shown in place of what it does. */
  blocked?: string
}

interface PersonActions {
  changeRole?: PersonAction
  remove?: PersonAction
}

/**
 * What an owner or admin may do to a person from People: change their role,
 * remove them. The server's rules decide (`membershipChangeBlockedReason`);
 * a refused action stays listed with the reason. The caller's own row offers
 * only a role change the server would accept. Nobody else gets either.
 */
export function personActions({
  member,
  context,
  members,
}: {
  member: OrganizationMember
  context: OrganizationContext | undefined
  members: readonly OrganizationMember[]
}): PersonActions {
  if (!canManage(context)) return {}
  if (member.status === "invited") {
    // Nothing to change yet; removing one withdraws the invitation.
    return {
      changeRole: {
        blocked: membershipChangeBlockedReason({ member, context, members }),
      },
      remove: member.invitation_id ? {} : undefined,
    }
  }
  const blocked = membershipChangeBlockedReason({ member, context, members })
  if (context?.organization_member_id === member.organization_member_id) {
    return blocked ? {} : { changeRole: {} }
  }
  return { changeRole: { blocked }, remove: { blocked } }
}
