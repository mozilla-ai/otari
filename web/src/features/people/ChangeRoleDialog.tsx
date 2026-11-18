import { useState } from "react"
import type { MembershipRole } from "@/client"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { Select } from "@/design-system/forms/Select"
import {
  asMembershipRole,
  isOwner,
  MEMBERSHIP_ROLE_OPTIONS,
} from "@/features/organization/roles"
import type { PersonRow } from "@/features/people/people"
import {
  useOrganizationContext,
  useUpdateOrganizationMember,
} from "@/shared/api/organizations"

/**
 * Change role: the organization role Members & roles sets from its Role
 * column, with the same write. Only an owner may make someone an owner, so
 * that choice is shown and disabled for an admin.
 */
export function ChangeRoleDialog({
  person,
  onClose,
}: {
  person: PersonRow
  onClose: () => void
}) {
  const update = useUpdateOrganizationMember()
  const isCallerOwner = isOwner(useOrganizationContext().data)
  const [role, setRole] = useState<MembershipRole>(
    asMembershipRole(person.role) ?? "member",
  )
  const memberId = person.member.organization_member_id
  return (
    <FormDialog
      isOpen
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      size="sm"
      title={`${person.name}’s role`}
      description="Owners and admins manage the organization; members and viewers see the workspaces they belong to."
      submitLabel="Change role"
      isPending={update.isPending}
      error={update.error}
      isDirty={role !== person.role}
      isSubmitDisabled={role === person.role || !memberId}
      onSubmit={() => {
        if (!memberId) return
        update.mutate({ id: memberId, body: { role } }, { onSuccess: onClose })
      }}
    >
      <Select
        label="Role"
        value={role}
        onChange={(value) => setRole(asMembershipRole(value) ?? role)}
        options={MEMBERSHIP_ROLE_OPTIONS.map((option) => ({
          ...option,
          isDisabled: option.value === "owner" && !isCallerOwner,
        }))}
      />
    </FormDialog>
  )
}
