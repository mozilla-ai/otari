import { Button, Card } from "@heroui/react"
import { useState } from "react"

import {
  useCreateOrganization,
  useSwitchOrganization,
} from "@/shared/api/hooks"
import { Field } from "@/shared/components/Field"
import { ErrorBanner } from "@/shared/components/ui"

// Create an organization and move into it. Two calls, because the server keeps
// them apart on purpose: creating one does not change which organization the
// rest of the session is looking at, and an operator setting one up for
// somebody else should not be moved out of their own. From the scope switcher
// the two belong together, so this chains them, and a switch that fails leaves
// the organization created and reachable from the same menu rather than lost.
export function CreateOrganizationForm({ onClose }: { onClose: () => void }) {
  const create = useCreateOrganization()
  const switchTo = useSwitchOrganization()
  const [name, setName] = useState("")
  const trimmed = name.trim()
  return (
    <Card>
      <Card.Content className="flex flex-col gap-4 p-5">
        <div className="text-title">Create organization</div>
        <ErrorBanner error={create.error ?? switchTo.error} />
        <Field
          label="Name"
          value={name}
          onChange={setName}
          placeholder="Research"
          isRequired
          autoFocus
          description="You become its owner, and it starts with a default workspace. Names do not have to be unique."
        />
        <div className="flex gap-2">
          <Button
            variant="primary"
            isDisabled={trimmed === ""}
            isPending={create.isPending || switchTo.isPending}
            onPress={() =>
              create.mutate(
                { name: trimmed },
                {
                  onSuccess: (organization) =>
                    switchTo.mutate(organization.id, { onSuccess: onClose }),
                },
              )
            }
          >
            Create organization
          </Button>
          <Button variant="ghost" onPress={onClose}>
            Cancel
          </Button>
        </div>
      </Card.Content>
    </Card>
  )
}
