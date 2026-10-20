import { useState } from "react"

import { useDirtySnapshot } from "@/design-system/forms/useDirtySnapshot"

import { BackupList } from "./BackupList"
import { PolicyGroup } from "./PolicyGroup"
import {
  buildSpec,
  isAtModelCap,
  isDraftReady,
  type PolicyDraft,
} from "./policyDraft"
import { MAX_CANDIDATES, type RoutingRow } from "./policyModel"
import { usePolicyWrite } from "./usePolicyWrite"

/** The backups a policy falls over to, in order: its `on_failure` chain. */
export function FallbackChainGroup({
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
  const [backups, setBackups] = useState(stored.backups)
  const { isDirty, reset } = useDirtySnapshot(backups)
  const draft = { ...stored, backups }

  return (
    <PolicyGroup
      title="Fallback chain"
      saveLabel="Save fallback chain"
      isDirty={isDirty}
      canSave={isDraftReady(draft)}
      isPending={writer.isPending}
      error={writer.error}
      onSave={() => writer.write(buildSpec(draft), { onSuccess: reset })}
    >
      <BackupList
        backups={backups}
        onChange={setBackups}
        isAtCap={isAtModelCap(draft)}
        maxModels={MAX_CANDIDATES}
      />
    </PolicyGroup>
  )
}
