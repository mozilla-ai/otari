/**
 * Whether an organization's guardrails are actually running, and what it
 * costs when one is not.
 *
 * `build_state` belongs to a definition, but a definition serves no traffic:
 * its mandates do. So the fact is reported on the definition and its
 * consequence on each mandate that points at it, and the two are computed here
 * so both rows read one rule.
 *
 * The consequence follows the request path (`src/gateway/services/guardrails.py`):
 * a mandate whose definition this worker does not hold is unevaluable, and an
 * unevaluable mandate refuses the request only when it is `block` **and**
 * `on_unavailable` is `block`. Every other one serves the request unchecked,
 * which is correct and silent, and is why it is the case this page shouts about.
 */

import type {
  OrganizationGuardrail,
  OrganizationGuardrailDefinition,
} from "@/client"

export type DefinitionHealth =
  | "running"
  | "starting"
  | "not_running"
  | "off"
  | "credentials_unreadable"

/** "" when the mandate serves as configured. */
export type MandateConsequence = "" | "refused" | "unchecked"

export function definitionHealth(
  definition: Pick<
    OrganizationGuardrailDefinition,
    "build_state" | "secrets_decryptable"
  >,
): DefinitionHealth {
  // Its own state because the fix differs: type the key again, or restore the
  // OTARI_SECRET_KEY it was saved under, rather than look at the log.
  if (
    !definition.secrets_decryptable &&
    definition.build_state !== "disabled"
  ) {
    return "credentials_unreadable"
  }
  switch (definition.build_state) {
    case "built":
      return "running"
    // It answers for one worker, a write's own response already carries the
    // rebuilt state, and it clears within the refresh interval.
    case "pending":
      return "starting"
    case "disabled":
      return "off"
    default:
      return "not_running"
  }
}

export function mandateConsequence(
  mandate: OrganizationGuardrail,
  definitions: readonly OrganizationGuardrailDefinition[],
): MandateConsequence {
  // A mandate that is off is skipped outright, and a remote one has no build
  // to report: whether its service answers is a fact about each request.
  if (!mandate.enabled || !mandate.definition_id) return ""
  const definition = definitions.find((row) => row.id === mandate.definition_id)
  if (definition === undefined) return ""
  const health = definitionHealth(definition)
  if (health === "running" || health === "starting") return ""
  return mandate.mode === "block" && mandate.on_unavailable === "block"
    ? "refused"
    : "unchecked"
}

/** How many mandates are serving requests unchecked right now. */
export function countServedUnchecked(
  mandates: readonly OrganizationGuardrail[],
  definitions: readonly OrganizationGuardrailDefinition[],
): number {
  return mandates.filter(
    (mandate) => mandateConsequence(mandate, definitions) === "unchecked",
  ).length
}
