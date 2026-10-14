import { useState } from "react"

import { useDirtySnapshot } from "@/design-system/forms/useDirtySnapshot"

import { GuardrailsSection } from "./GuardrailsSection"
import { PolicyGroup } from "./PolicyGroup"
import { buildSpec, isDraftReady, type PolicyDraft } from "./policyDraft"
import type { RoutingRow } from "./policyModel"
import { useGuardrailsConfigured } from "./useGuardrailsConfigured"
import { usePolicyWrite } from "./usePolicyWrite"

/** The checks every request through this policy runs. */
export function GuardrailsGroup({
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
  const isConfigured = useGuardrailsConfigured(true)
  const [guardrails, setGuardrails] = useState(stored.guardrails)
  const { isDirty, reset } = useDirtySnapshot(guardrails)
  const draft = { ...stored, guardrails }

  return (
    <PolicyGroup
      title="Guardrails"
      saveLabel="Save guardrails"
      isDirty={isDirty}
      canSave={isDraftReady(draft)}
      isPending={writer.isPending}
      error={writer.error}
      onSave={() => writer.write(buildSpec(draft), { onSuccess: reset })}
    >
      <GuardrailsSection
        guardrails={guardrails}
        onChange={setGuardrails}
        isConfigured={isConfigured}
      />
    </PolicyGroup>
  )
}
