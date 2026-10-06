import type {
  BuiltInGuardrailCatalog,
  OrganizationGuardrailDefinition,
} from "@/client"
import {
  GuardrailTestDialog,
  shownTestError,
} from "@/features/guardrails/GuardrailTestDialog"
import { useTestOrganizationGuardrailDefinition } from "@/shared/api/guardrails"

/**
 * Test one definition's guardrail. It runs what is already built, so it
 * answers "does what is running work" rather than building anything.
 */
export function DefinitionTestDialog({
  isOpen,
  onClose,
  definition,
  catalog,
}: {
  isOpen: boolean
  onClose: () => void
  definition: OrganizationGuardrailDefinition
  catalog: BuiltInGuardrailCatalog | undefined
}) {
  const test = useTestOrganizationGuardrailDefinition()
  const specs =
    catalog?.guardrails?.find(
      (spec) => spec.guardrail_name === definition.guardrail_name,
    )?.validate_parameters ?? []
  return (
    <GuardrailTestDialog
      isOpen={isOpen}
      onClose={onClose}
      name={definition.name}
      specs={specs}
      isDescribed
      stored={undefined}
      identity={definition.id}
      extraJsonDescription="Handed to the guardrail with the check, as a mandate would."
      onRun={(body) => test.mutate({ definitionId: definition.id, body })}
      verdict={test.data}
      isPending={test.isPending}
      error={shownTestError(test.error, "the vendor call failed")}
    />
  )
}
