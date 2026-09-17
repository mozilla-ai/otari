/**
 * Turning the built-in guardrail catalog into the two controls that choose one.
 *
 * An operator knows what they want checked long before they know which vendor
 * checks it, so the form asks for the operation first and offers only the
 * guardrails that do it.
 *
 * Nothing about the taxonomy is written down here. The set of operations, which
 * guardrails do each one, and the order they are offered in are all derived from
 * `GET /tool-settings/guardrails/catalog`, which publishes any-guardrail's own
 * metadata unchanged. A category upstream adds therefore reaches these controls
 * with no edit in this repository.
 *
 * Two decisions that derivation does not make on its own:
 *
 * **`categories`, not `primary_category`.** Upstream's own grouping helper,
 * `AnyGuardrail.group_by("category")`, documents that for a set-valued dimension
 * "a guardrail appears under every value it carries", and `primary_category`
 * exists to place a guardrail exactly once in grouped docs navigation, which is
 * not what a picker does. Reading the headline alone would hide Alinia, Patronus
 * and watsonx Guardian from prompt injection although all three detect it.
 *
 * **Sorted.** That is what `group_by` returns, and it is the only order that does
 * not move when the catalog grows.
 */

import type { BuiltInGuardrailSpec } from "@/client"
import type { SelectOption } from "@/design-system/forms/Select"
import {
  configurableFor,
  jsonFieldSpec,
  mappedFor,
} from "@/features/tools/guardrailFieldSuggestions"
import { parameterLabel } from "@/features/tools/guardrailParameters"

/**
 * Spelled-out labels for wire names the mechanical rule cannot expand.
 *
 * Formatting, not taxonomy. Which operations exist is still read from the
 * catalog and nothing here adds or removes one: a name absent from this map
 * falls through to `parameterLabel`, so a category upstream adds still renders,
 * as "Something new" rather than as nothing.
 *
 * Only acronyms belong here. `prompt_injection` becomes "Prompt injection" on
 * its own, but `pii` becomes "Pii", which reads as a typo beside eight labels
 * that are words.
 */
const SPELLED_OUT: Record<string, string> = {
  pii: "Personally Identifiable Information",
}

/**
 * Jobs the first control does not offer.
 *
 * `general_judge` is not a thing to detect. It is "write the rule yourself and
 * have a model decide", which is a different kind of answer to "what do you
 * want checked?" than the eight subjects beside it, and an operator who wants
 * it is not shopping for a detector.
 *
 * The one exclusion in this module, and the only place a category name is
 * written down. It costs the guardrails whose every category is in it: AnyLlm
 * declares nothing else and is therefore not selectable here.
 */
const NOT_OFFERED = new Set(["general_judge"])

/**
 * Whether this guardrail is offered for that job.
 *
 * Two ways to qualify, because the catalog's `categories` is neither the whole
 * answer nor always right. A documented key that asks the guardrail for the job
 * settles it, which is how watsonx Guardian reaches personal data with no `pii`
 * category on it. Otherwise the category has to be declared *and* reachable:
 * Alinia declares personal data and nothing names the detection that does it,
 * so offering it would be offering a dead end.
 */
function offersOperation(
  spec: BuiltInGuardrailSpec,
  operation: string,
): boolean {
  if (NOT_OFFERED.has(operation)) return false
  const parameters = spec.create_parameters ?? []
  if (mappedFor(spec.guardrail_name, parameters, operation)) return true
  return (
    spec.categories.includes(operation) &&
    configurableFor(spec.guardrail_name, parameters, operation)
  )
}

/** Every job one guardrail is offered for. */
function offeredOperations(spec: BuiltInGuardrailSpec): string[] {
  const mapped = (spec.create_parameters ?? []).flatMap((parameter) =>
    Object.keys(
      jsonFieldSpec(spec.guardrail_name, parameter)?.byCategory ?? {},
    ),
  )
  return [...new Set([...spec.categories, ...mapped])].filter((category) =>
    offersOperation(spec, category),
  )
}

/** What an operation is called in the dashboard. */
export function operationLabel(category: string): string {
  return SPELLED_OUT[category] ?? parameterLabel(category)
}

/**
 * Every operation the catalog can do, sorted.
 *
 * An operation no installed guardrail names is left out rather than offered
 * empty: an operator who picked it would reach a second control with nothing in
 * it and no reason given.
 */
export function operationOptions(
  guardrails: readonly BuiltInGuardrailSpec[],
): SelectOption[] {
  const present = new Set<string>()
  for (const spec of guardrails) {
    for (const category of offeredOperations(spec)) present.add(category)
  }
  return [...present].sort().map((category) => ({
    value: category,
    label: operationLabel(category),
  }))
}

/** The guardrails that do one operation, by the name they are shown under. */
export function guardrailsForOperation(
  guardrails: readonly BuiltInGuardrailSpec[],
  operation: string,
): BuiltInGuardrailSpec[] {
  if (operation === "") return []
  return guardrails
    .filter((spec) => offersOperation(spec, operation))
    .sort((a, b) => a.display_name.localeCompare(b.display_name))
}

/**
 * The second control's rows.
 *
 * `SelectOption` carries no second line, so the vendor rides on the label: two
 * guardrails doing one job are told apart by who publishes them far more often
 * than by their own names. What the chosen one actually does is its own
 * `description`, shown under the control once there is one to show.
 */
export function guardrailOptions(
  guardrails: readonly BuiltInGuardrailSpec[],
  operation: string,
): SelectOption[] {
  return guardrailsForOperation(guardrails, operation).map((spec) => ({
    value: spec.guardrail_name,
    label: `${spec.display_name} · ${spec.vendor}`,
  }))
}

/**
 * Where any-guardrail documents one guardrail.
 *
 * Derived rather than tabulated, like everything else here: the reference files
 * a guardrail under its headline category and names the page after the class,
 * both in kebab case, so `lakera_guard` under `prompt_injection` is
 * `prompt-injection/lakera-guard`. Checked against the published index for all
 * nine guardrails this gateway can run.
 *
 * `primary_category` is the right field for once, and the one place in this
 * module that is not `categories`: the picker asks which guardrails do a job, so
 * a guardrail belongs under every category it detects, but a documentation site
 * files each page once and the headline is where it filed this one.
 *
 * A page that has moved gives a 404 on a site nobody here controls, which is why
 * this is a link beside the description and never the description itself.
 */
export function guardrailDocsHref(spec: BuiltInGuardrailSpec): string {
  const slug = (value: string) => value.replaceAll("_", "-")
  return `https://docs.mozilla.ai/any-guardrail/api-reference/index/${slug(spec.primary_category)}/${slug(spec.guardrail_name)}`
}

/** One guardrail's row in the catalog, or undefined when nothing names it. */
export function findGuardrail(
  guardrails: readonly BuiltInGuardrailSpec[],
  guardrailName: string,
): BuiltInGuardrailSpec | undefined {
  return guardrails.find((spec) => spec.guardrail_name === guardrailName)
}

/**
 * A name for a new definition, taken from the operation it does.
 *
 * The name is what a caller sends as its `profile`, and "prompt-injection" is
 * what it would have been typed as anyway, so suggesting it leaves the ordinary
 * case with one field to fill: the credential.
 */
export function suggestName(
  operation: string,
  taken: readonly string[],
): string {
  if (operation === "") return ""
  const base = operation.replaceAll("_", "-")
  const used = new Set(taken)
  if (!used.has(base)) return base
  let suffix = 2
  while (used.has(`${base}-${suffix}`)) suffix += 1
  return `${base}-${suffix}`
}
