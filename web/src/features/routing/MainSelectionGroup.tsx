import { useState } from "react"

import { useDirtySnapshot } from "@/design-system/forms/useDirtySnapshot"
import { Segmented } from "@/design-system/navigation/Segmented"
import { OfferingPicker } from "@/features/models/OfferingPicker"

import { PolicyGroup } from "./PolicyGroup"
import {
  buildSpec,
  isAtModelCap,
  isDraftReady,
  type PolicyDraft,
  SELECTIONS,
  type Selection,
  selectionOf,
  switchSelection,
} from "./policyDraft"
import type { RoutingRow } from "./policyModel"
import { RoutedPoolSection } from "./RoutedPoolSection"
import { usePolicyWrite } from "./usePolicyWrite"

/** What serves a request before any backup: one model, or a routed pool. */
export function MainSelectionGroup({
  row,
  stored,
  isDeploymentWide,
  fallbackWorkspaceId,
}: {
  row: RoutingRow
  /** The policy as saved; every other section is written back as this. */
  stored: PolicyDraft
  isDeploymentWide: boolean
  fallbackWorkspaceId: string | null
}) {
  const writer = usePolicyWrite(row, { isDeploymentWide, fallbackWorkspaceId })
  const [main, setMain] = useState({ target: stored.target, pool: stored.pool })
  const draft = { ...stored, ...main }
  // Re-seeded on a save, so the section reads clean before the refetch lands.
  const { isDirty, reset } = useDirtySnapshot(main)
  const isAlias = row.kind === "alias"

  return (
    <PolicyGroup
      title="Main selection"
      description="What serves a request before any backup is tried."
      saveLabel="Save main selection"
      isDirty={isDirty}
      canSave={isDraftReady(draft)}
      isPending={writer.isPending}
      error={writer.error}
      onSave={() => writer.write(buildSpec(draft), { onSuccess: reset })}
    >
      {isAlias ? null : (
        <Segmented
          label="How the main model is chosen"
          options={SELECTIONS.map((option) => ({ ...option }))}
          value={selectionOf(draft)}
          onChange={(next) => {
            const { target, pool } = switchSelection(draft, next as Selection)
            setMain({ target, pool })
          }}
        />
      )}
      {selectionOf(draft) === "single" ? (
        <OfferingPicker
          id="main-model"
          label="Main model"
          value={main.target}
          onChange={(target) => setMain((prev) => ({ ...prev, target }))}
          isRequired
          description="Serves every request first. Callers never see it."
        />
      ) : (
        <RoutedPoolSection
          pool={main.pool}
          onChange={(pool) => setMain((prev) => ({ ...prev, pool }))}
          isAtCap={isAtModelCap(draft)}
        />
      )}
    </PolicyGroup>
  )
}
