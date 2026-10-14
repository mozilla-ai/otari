import { useState } from "react"

import { PolicyGroup } from "./PolicyGroup"
import { nameProblemOf, PolicyNameField } from "./PolicyNameField"
import type { RoutingRow } from "./policyModel"
import { usePolicyWrite } from "./usePolicyWrite"

/** The policy's name, which a rename moves rather than copies. */
export function PolicyNameGroup({
  row,
  isDeploymentWide,
  fallbackWorkspaceId,
  onRenamed,
}: {
  row: RoutingRow
  isDeploymentWide: boolean
  fallbackWorkspaceId: string | null
  onRenamed: (name: string) => void
}) {
  const writer = usePolicyWrite(row, { isDeploymentWide, fallbackWorkspaceId })
  const [name, setName] = useState(row.name)
  const next = name.trim()

  // The alias API has no rename, so an alias's name stays fixed.
  if (row.kind === "alias")
    return (
      <PolicyGroup title="Name" isReadOnly>
        <div className="flex flex-col gap-1">
          <code className="text-sm">{row.name}</code>
          <span className="text-caption">
            An alias name is its key and cannot be changed here. Delete and
            recreate it to change it.
          </span>
        </div>
      </PolicyGroup>
    )

  return (
    <PolicyGroup
      title="Name"
      saveLabel="Rename"
      isDirty={next !== "" && next !== row.name}
      canSave={nameProblemOf(name, row.name) === ""}
      isPending={writer.isPending}
      error={writer.error}
      onSave={() =>
        writer.write(row.spec, {
          name: next,
          onSuccess: () => onRenamed(next),
        })
      }
    >
      <PolicyNameField
        value={name}
        onChange={setName}
        previousName={row.name}
      />
    </PolicyGroup>
  )
}
