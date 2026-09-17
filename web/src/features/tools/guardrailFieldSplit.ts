/**
 * Which of a guardrail's arguments an operator has to decide, and which have
 * been decided for them.
 *
 * `required` alone does not answer this, and reading it as though it did is what
 * put watsonx Guardian's API key behind an accordion: nothing on that guardrail
 * is `required`, because its credentials sit in one-of requirement groups
 * instead. It also leaves Bedrock's AWS keys hidden, which are `secret` and have
 * no default and are plainly the first thing anyone fills in.
 *
 * So four tests, any of which puts a field on the page. Everything else is a
 * value the vendor already chose or a detail for the deployment that wants it,
 * and belongs behind a press.
 */

import type { BuiltInGuardrailSpec, GuardrailParameterSpec } from "@/client"
import { jsonFieldSpec } from "@/features/tools/guardrailFieldSuggestions"

/** Every parameter named by a one-of requirement group. */
function inAnyGroup(spec: BuiltInGuardrailSpec): Set<string> {
  const named = new Set<string>()
  for (const group of spec.requirement_groups ?? []) {
    for (const parameter of group.parameters) named.add(parameter)
  }
  return named
}

/**
 * Whether this argument is the operator's to answer.
 *
 * 1. **Required.** The signature gives no default, so there is nothing to fall
 *    back to.
 * 2. **A credential.** Always a decision, and a deployment that has it in an
 *    environment variable is still better served by seeing the field than by
 *    hunting for it.
 * 3. **In a requirement group.** One of the group must be supplied, which no
 *    member's own `required` flag can say. This is the whole of watsonx.
 * 4. **The guardrail's own subject.** A field the suggestions describe with
 *    options is what the guardrail actually checks: watsonx's detectors, Alinia's
 *    detections, Patronus's evaluators. Hiding it would hide the point.
 *
 * An argument that cannot be stored is never a decision: there is no value to
 * give it, and it renders disabled saying so.
 */
export function isDecision(
  guardrail: BuiltInGuardrailSpec,
  parameter: GuardrailParameterSpec,
  grouped: Set<string> = inAnyGroup(guardrail),
): boolean {
  if (parameter.storable === false) return false
  if (parameter.required) return true
  if (parameter.secret) return true
  if (grouped.has(parameter.name)) return true
  return Boolean(jsonFieldSpec(guardrail.guardrail_name, parameter)?.options)
}

/** One stage's parameters, split into what is shown and what is folded away. */
export function splitParameters(
  guardrail: BuiltInGuardrailSpec | undefined,
  parameters: readonly GuardrailParameterSpec[],
): { decisions: GuardrailParameterSpec[]; extras: GuardrailParameterSpec[] } {
  // Nothing known, so nothing may be called advanced: hiding a field because
  // this build cannot describe its guardrail is the one failure this split must
  // not have.
  if (!guardrail) return { decisions: [...parameters], extras: [] }
  const grouped = inAnyGroup(guardrail)
  const decisions: GuardrailParameterSpec[] = []
  const extras: GuardrailParameterSpec[] = []
  for (const parameter of parameters) {
    if (isDecision(guardrail, parameter, grouped)) decisions.push(parameter)
    else extras.push(parameter)
  }
  return { decisions, extras }
}
