import { useState } from "react"
import { FormDialog } from "@/design-system/feedback/FormDialog"
import { Field } from "@/design-system/forms/Field"
import {
  useCreateOrganization,
  useSwitchOrganization,
} from "@/shared/api/organizations"

// Create an organization and move into it. Two calls, because the server keeps
// them apart on purpose: creating one does not change which organization the
// rest of the session is looking at, and an operator setting one up for
// somebody else should not be moved out of their own. From the scope switcher
// the two belong together, so this chains them, and a switch that fails leaves
// the organization created and reachable from the same menu rather than lost.
export function CreateOrganizationForm({
  isOpen,
  onClose,
}: {
  isOpen: boolean
  onClose: () => void
}) {
  const create = useCreateOrganization()
  const switchTo = useSwitchOrganization()
  const [name, setName] = useState("")
  const trimmed = name.trim()
  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      // One field, which is what `sm` is for.
      size="sm"
      title="New organization"
      submitLabel="Create organization"
      onSubmit={() =>
        create.mutate(
          { name: trimmed },
          {
            onSuccess: (organization) =>
              switchTo.mutate(organization.id, { onSuccess: onClose }),
          },
        )
      }
      isPending={create.isPending || switchTo.isPending}
      isSubmitDisabled={trimmed === ""}
      isDirty={trimmed !== ""}
      error={create.error ?? switchTo.error}
    >
      <Field
        label="Name"
        value={name}
        onChange={setName}
        placeholder="Research"
        isRequired
        autoFocus
        description="You become its owner, and it starts with a default workspace. Names do not have to be unique."
        reserveMessage
      />
    </FormDialog>
  )
}
