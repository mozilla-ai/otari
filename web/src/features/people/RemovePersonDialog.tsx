import { useNavigate } from "@tanstack/react-router"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import type { PersonRow } from "@/features/people/people"
import {
  useRemoveOrganizationMember,
  useRevokeOrganizationMemberInvitation,
} from "@/shared/api/organizations"

/**
 * Remove from organization: a member's membership is suspended, as Members &
 * roles does it; an invitation not yet accepted is revoked. Either way the
 * person leaves People, and the page goes back to the list.
 */
export function RemovePersonDialog({
  person,
  isOpen,
  onOpenChange,
}: {
  person: PersonRow
  isOpen: boolean
  onOpenChange: (open: boolean) => void
}) {
  const navigate = useNavigate()
  const remove = useRemoveOrganizationMember()
  const revoke = useRevokeOrganizationMemberInvitation()
  const memberId = person.member.organization_member_id
  const invitationId = person.member.invitation_id
  const done = {
    onSuccess: () => {
      onOpenChange(false)
      void navigate({ to: "/people" })
    },
  }
  return (
    <ConfirmDialog
      isOpen={isOpen}
      onOpenChange={onOpenChange}
      heading={`Remove ${person.name}?`}
      body={
        person.isInvited
          ? "Their invitation stops working, and they leave this list."
          : "They leave this list. What they already ran and spent stays on record, and inviting the same address again restores the same membership."
      }
      confirmLabel="Remove from organization"
      isPending={remove.isPending || revoke.isPending}
      error={remove.error ?? revoke.error}
      onConfirm={() => {
        if (person.isInvited && invitationId) revoke.mutate(invitationId, done)
        else if (memberId) remove.mutate(memberId, done)
      }}
    />
  )
}
