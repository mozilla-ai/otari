import { useState } from "react"
import type { Organization } from "@/client"
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
  // Only the second call is left to retry once the first has succeeded, so the
  // organization it returned is held here and the submit becomes that call.
  // Pressing the button again otherwise creates a second organization, which
  // is the one failure of this pair an operator cannot undo from the menu.
  const [created, setCreated] = useState<Organization | null>(null)
  const trimmed = name.trim()
  return (
    <FormDialog
      isOpen={isOpen}
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      // One field, which is what `sm` is for.
      size="sm"
      // The state, once there is one: nothing is left to create, and the frame
      // saying "New organization" over a created one is the last thing still
      // claiming nothing happened. Same shape as the keys page's secret step.
      title={created ? "Organization created" : "New organization"}
      submitLabel={created ? "Switch to organization" : "Create organization"}
      onSubmit={() => {
        if (created) {
          switchTo.mutate(created.id, { onSuccess: onClose })
          return
        }
        create.mutate(
          { name: trimmed },
          {
            onSuccess: (organization) => {
              setCreated(organization)
              switchTo.mutate(organization.id, { onSuccess: onClose })
            },
          },
        )
      }}
      isPending={create.isPending || switchTo.isPending}
      isSubmitDisabled={created === null && trimmed === ""}
      // Closing after the create succeeded discards nothing: the organization
      // exists, and the menu it was started from lists it.
      isDirty={created === null && trimmed !== ""}
      error={create.error ?? switchTo.error}
    >
      {created ? (
        // The success step swaps the children as well as the label: a form that
        // still looks unsubmitted reads as "nothing happened", and the operator
        // closes it and creates the organization a second time. The field is
        // gone rather than disabled, and so is its description, which describes
        // a create that has already happened.
        <p className="text-body">
          {switchTo.isPending
            ? `${created.name} was created. Switching into it…`
            : `${created.name} was created. Switching into it failed.`}
        </p>
      ) : (
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
      )}
    </FormDialog>
  )
}
