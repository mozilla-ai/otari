import { useState } from "react"

import { useDirtySnapshot } from "@/design-system/forms/useDirtySnapshot"

import { PolicyGroup } from "./PolicyGroup"
import { buildSpec, isDraftReady, type PolicyDraft } from "./policyDraft"
import type { RoutingRow } from "./policyModel"
import { TierDownSection } from "./TierDownSection"
import { usePolicyWrite } from "./usePolicyWrite"

/** The cheaper model a policy switches to once its budget fills. */
export function TierDownGroup({
  row,
  stored,
  isDeploymentWide,
  fallbackWorkspaceId,
}: {
  row: RoutingRow
  stored: PolicyDraft
  isDeploymentWide: boolean
  fallbackWorkspaceId: string | null
}) {
  const writer = usePolicyWrite(row, { isDeploymentWide, fallbackWorkspaceId })
  const [tierDowns, setTierDowns] = useState(stored.tierDowns)
  const { isDirty, reset } = useDirtySnapshot(tierDowns)
  const draft = { ...stored, tierDowns }

  return (
    <PolicyGroup
      title="Budget tier-down"
      saveLabel="Save budget tier-down"
      isDirty={isDirty}
      canSave={isDraftReady(draft)}
      isPending={writer.isPending}
      error={writer.error}
      onSave={() => writer.write(buildSpec(draft), { onSuccess: reset })}
    >
      <TierDownSection tierDowns={tierDowns} onChange={setTierDowns} />
    </PolicyGroup>
  )
}
