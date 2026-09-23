/**
 * Reading the built-in guardrail catalog for the definition form: which checks
 * to offer, which guardrails perform each one, and in what order a chosen
 * guardrail's fields are asked for.
 *
 * The catalog is `GET /v1/tool-settings/guardrails/catalog`, the guardrails the
 * installed any-guardrail library can build and call over a hosted API. It says
 * what is possible; what an organization may store is the definition store's
 * answer, which is why one filter below lives here and not in the catalog.
 */

import type {
  BuiltInGuardrailCatalog,
  BuiltInGuardrailSpec,
  GuardrailParameterSpec,
} from "@/client"
import { jsonFieldSpec } from "@/features/guardrails/guardrailFieldSuggestions"
import { parameterLabel } from "@/features/guardrails/guardrailParameters"

// Mirrors `_NOT_DEFINABLE_BY_AN_ORGANIZATION` in
// `src/gateway/services/guardrail_catalog.py`. `any_llm` builds with no
// credential and judges text with the process's own LLM key, so an
// organization storing one would spend the operator's key unmetered. The
// catalog lists it on purpose, a deployment-wide store may one day take it,
// and the organization store refuses it; offering it here would only lead to
// that refusal.
const NOT_DEFINABLE = new Set(["any_llm"])

// A category that names no check a user picks from a list. Patronus carries it
// beside four concrete checks, so dropping it hides no guardrail.
const NOT_A_CHECK = new Set(["general_judge"])

const CHECK_LABELS: Record<string, string> = {
  pii: "Personally identifiable information",
  off_topic: "Off-topic",
}

/** The guardrails an organization may define, in catalog order. */
export function definableGuardrails(
  catalog: BuiltInGuardrailCatalog | undefined,
): BuiltInGuardrailSpec[] {
  return (catalog?.guardrails ?? []).filter(
    (spec) => !NOT_DEFINABLE.has(spec.guardrail_name),
  )
}

/** A check as the picker says it. */
export function checkLabel(check: string): string {
  return CHECK_LABELS[check] ?? parameterLabel(check)
}

/**
 * Every check some definable guardrail performs, sorted.
 *
 * The union of each guardrail's `categories`, not its `primary_category`: a
 * guardrail's primary is one of several things it checks, and reading only
 * that would hide Alinia, Patronus and watsonx Guardian from prompt injection.
 */
export function guardrailChecks(
  catalog: BuiltInGuardrailCatalog | undefined,
): string[] {
  const checks = new Set<string>()
  for (const spec of definableGuardrails(catalog)) {
    for (const category of spec.categories) {
      if (!NOT_A_CHECK.has(category)) checks.add(category)
    }
  }
  return [...checks].sort()
}

/** The definable guardrails that perform one check, by display name. */
export function guardrailsForCheck(
  catalog: BuiltInGuardrailCatalog | undefined,
  check: string,
): BuiltInGuardrailSpec[] {
  return definableGuardrails(catalog)
    .filter((spec) => spec.categories.includes(check))
    .sort((a, b) => a.display_name.localeCompare(b.display_name))
}

export type CreateFieldLayout = {
  /** Drawn in the form's body. */
  fields: GuardrailParameterSpec[]
  /** Drawn inside Advanced. */
  advanced: GuardrailParameterSpec[]
  /** Arguments the form cannot store, named once inside Advanced. */
  unstorable: string[]
}

/**
 * Split a guardrail's constructor arguments into what the form asks first and
 * what waits behind Advanced.
 *
 * Secrets first, then members of an either/or group, then the other required
 * ones, then any set of checkboxes. Not required-first: watsonx Guardian marks
 * nothing required, because each of its credentials is one member of a group
 * whose other member is a live client object, so a required-first rule would
 * hide its API key.
 *
 * An argument the catalog marks unstorable (a live object, not configuration)
 * gets no control at all.
 */
export function createFieldLayout(
  spec: BuiltInGuardrailSpec | undefined,
): CreateFieldLayout {
  const parameters = spec?.create_parameters ?? []
  const grouped = new Set(
    (spec?.requirement_groups ?? []).flatMap((group) => group.parameters),
  )
  const storable = parameters.filter((parameter) => parameter.storable)
  const secrets = storable.filter((parameter) => parameter.secret)
  const members = storable.filter(
    (parameter) => !parameter.secret && grouped.has(parameter.name),
  )
  const required = storable.filter(
    (parameter) =>
      !parameter.secret && !grouped.has(parameter.name) && parameter.required,
  )
  // A switch set the form draws as checkboxes is what the guardrail is told to
  // check, whatever the catalog's required flag says: watsonx Guardian marks
  // its detectors optional and runs nothing without one.
  const switches = storable.filter(
    (parameter) =>
      !parameter.secret &&
      !grouped.has(parameter.name) &&
      !parameter.required &&
      isSwitchSet(spec, parameter),
  )
  const fields = [...secrets, ...members, ...required, ...switches]
  const asked = new Set(fields.map((parameter) => parameter.name))
  return {
    fields,
    advanced: storable.filter((parameter) => !asked.has(parameter.name)),
    unstorable: parameters
      .filter((parameter) => !parameter.storable)
      .map((parameter) => parameter.name),
  }
}

function isSwitchSet(
  spec: BuiltInGuardrailSpec | undefined,
  parameter: GuardrailParameterSpec,
): boolean {
  if (spec === undefined || parameter.type !== "json") return false
  const kind = jsonFieldSpec(spec.guardrail_name, parameter)?.kind
  return kind === "flags" || kind === "presets"
}

/** A name for a new definition that the organization does not use yet. */
export function suggestDefinitionName(
  guardrailName: string,
  taken: readonly string[],
): string {
  const base = guardrailName.replaceAll("_", "-")
  const used = new Set(taken)
  if (!used.has(base)) return base
  let suffix = 2
  while (used.has(`${base}-${suffix}`)) suffix += 1
  return `${base}-${suffix}`
}
