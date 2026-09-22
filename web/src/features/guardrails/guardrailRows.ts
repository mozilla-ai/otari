/** What the organization guardrails page writes in each cell. */

import type {
  BuiltInGuardrailCatalog,
  OrganizationGuardrail,
  OrganizationGuardrailDefinition,
} from "@/client"
import { checkLabel } from "@/features/guardrails/builtInCatalog"
import { parameterLabel } from "@/features/guardrails/guardrailParameters"

// The catalog lists it beside real checks, and it names none a reader picks.
const NOT_A_CHECK = "general_judge"

function specFor(
  definition: OrganizationGuardrailDefinition,
  catalog: BuiltInGuardrailCatalog | undefined,
) {
  return catalog?.guardrails?.find(
    (spec) => spec.guardrail_name === definition.guardrail_name,
  )
}

/** The guardrail a definition builds: product, then maker. */
export function guardrailLabel(
  definition: OrganizationGuardrailDefinition,
  catalog: BuiltInGuardrailCatalog | undefined,
): string {
  const spec = specFor(definition, catalog)
  if (spec === undefined) return definition.guardrail_name
  return spec.vendor
    ? `${spec.display_name} · ${spec.vendor}`
    : spec.display_name
}

/** The checks a definition performs, in the picker's words. */
export function definitionChecks(
  definition: OrganizationGuardrailDefinition,
  catalog: BuiltInGuardrailCatalog | undefined,
): string {
  return (specFor(definition, catalog)?.categories ?? [])
    .filter((category) => category !== NOT_A_CHECK)
    .map(checkLabel)
    .sort()
    .join(", ")
}

/** Which secrets a definition holds, by name only. */
export function credentialsLabel(
  definition: OrganizationGuardrailDefinition,
): string {
  if (!definition.secrets_decryptable) return "unreadable"
  const names = Object.keys(definition.create_secrets)
  if (names.length === 0) return "none"
  return `${names.map(parameterLabel).join(", ")} set`
}

/**
 * Who runs a mandate's check: a definition by its name, the host of the
 * mandate's own endpoint, or the deployment's service when it names neither.
 */
export function runsOnLabel(
  mandate: OrganizationGuardrail,
  definitions: readonly OrganizationGuardrailDefinition[],
): string {
  if (mandate.definition_id) {
    return (
      definitions.find((row) => row.id === mandate.definition_id)?.name ??
      "a guardrail you set up"
    )
  }
  if (mandate.url) {
    try {
      return new URL(mandate.url).host
    } catch {
      return mandate.url
    }
  }
  return "the deployment guardrails service"
}

/**
 * What happens to a request when the check cannot run. A monitoring mandate
 * always serves it, so its stored choice is not what happens and is not shown.
 */
export function ifItCantRunLabel(mandate: OrganizationGuardrail): string {
  return mandate.mode === "block" && mandate.on_unavailable === "block"
    ? "Refuse the request"
    : "Serve it unchecked"
}

/** The mandates that run a definition. */
export function mandatesOn(
  definition: OrganizationGuardrailDefinition,
  mandates: readonly OrganizationGuardrail[],
): OrganizationGuardrail[] {
  return mandates.filter((mandate) => mandate.definition_id === definition.id)
}
