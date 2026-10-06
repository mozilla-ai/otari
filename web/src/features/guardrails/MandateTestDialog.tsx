import type { GuardrailCatalog, OrganizationGuardrail } from "@/client"
import {
  GuardrailTestDialog,
  shownTestError,
} from "@/features/guardrails/GuardrailTestDialog"
import {
  findProfile,
  parameterSpecs,
  profileIdentity,
} from "@/features/guardrails/guardrailParameters"
import { useTestOrganizationGuardrail } from "@/shared/api/tools"

/**
 * Test a mandate that runs on your own guardrails service: its endpoint and
 * credential, starting from its stored per-check arguments. A mandate on a
 * configured guardrail is tested from that guardrail's row instead.
 */
export function MandateTestDialog({
  isOpen,
  onClose,
  mandate,
  catalog,
}: {
  isOpen: boolean
  onClose: () => void
  mandate: OrganizationGuardrail
  catalog: GuardrailCatalog | undefined
}) {
  const test = useTestOrganizationGuardrail()
  return (
    <GuardrailTestDialog
      isOpen={isOpen}
      onClose={onClose}
      name={mandate.profile}
      specs={parameterSpecs(catalog, mandate.profile)}
      isDescribed={findProfile(catalog, mandate.profile) !== undefined}
      stored={mandate.validate_kwargs}
      identity={profileIdentity(catalog, mandate.profile)}
      extraJsonDescription="Sent with the check in place of what is stored, for this test only."
      onRun={(body) => test.mutate({ guardrailId: mandate.id, body })}
      verdict={test.data}
      isPending={test.isPending}
      error={shownTestError(
        test.error,
        "the guardrails service could not be reached",
      )}
    />
  )
}
