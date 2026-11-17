import { Skeleton } from "@/design-system/feedback/Skeleton"
import { MemberAvatar } from "@/features/account/avatar/MemberAvatar"
import { memberLabel } from "@/features/organization/roles"
import { useOrganizationMembers } from "@/shared/api/organizations"

/**
 * Who a key bills to, as the Sessions table shows a person: avatar and name.
 * An owner who is not on the roster (removed, or a deployment user who never
 * joined) is shown by the id the key bills to, once the roster has been read.
 */
export function KeyOwner({ userId }: { userId: string | null }) {
  const members = useOrganizationMembers()
  const member = userId
    ? (members.data ?? []).find(
        (one) => one.attribution_user_id === userId || one.user_id === userId,
      )
    : undefined
  if (!members.data) {
    return members.isError ? (
      <span className="text-sm text-subtle">—</span>
    ) : (
      <Skeleton className="h-2.5 w-30 max-w-full" />
    )
  }
  if (!member) {
    return (
      <span
        className="block truncate text-mono-caption text-subtle"
        title="Not in the organization"
      >
        {userId ?? "—"}
      </span>
    )
  }
  return (
    <span className="flex min-w-0 items-center gap-2">
      <MemberAvatar
        userId={member.user_id ?? undefined}
        face={member.avatar_face}
        isCompact
      />
      <span
        className="truncate text-sm text-foreground"
        title={memberLabel(member)}
      >
        {memberLabel(member)}
      </span>
    </span>
  )
}
