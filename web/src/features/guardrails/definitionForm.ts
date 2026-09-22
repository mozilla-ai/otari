/**
 * The secret round trip for a guardrail definition's constructor arguments.
 *
 * A definition answers with its plain arguments in `create_kwargs` and each
 * stored secret as `***` under its own name in `create_secrets`. On a write,
 * `create_kwargs` replaces the arguments whole: a secret sent as `***` keeps its
 * stored value, a new value replaces it, and a secret **left out is cleared**.
 * That last rule is the opposite of a mandate's `credential`, where leaving the
 * field out keeps it, so the two forms cannot share a builder.
 */

import type {
  GuardrailParameterSpec,
  OrganizationGuardrailDefinition,
} from "@/client"
import {
  buildValidateKwargs,
  type ParameterValues,
} from "@/features/guardrails/guardrailParameters"
import { REDACTED_SECRET } from "@/shared/helpers/redaction"

/** The secret names a stored definition holds, and none for a new one. */
export function heldSecrets(
  definition:
    | Pick<OrganizationGuardrailDefinition, "create_secrets">
    | undefined,
): Set<string> {
  return new Set(Object.keys(definition?.create_secrets ?? {}))
}

/**
 * What the form seeds from: the stored arguments without any secret. Every
 * secret starts blank, because a password box holding the mask shows three
 * real-looking characters, which a user cannot tell from "cleared".
 */
export function seedableArguments(
  specs: GuardrailParameterSpec[],
  definition:
    | Pick<OrganizationGuardrailDefinition, "create_kwargs">
    | undefined,
): Record<string, unknown> {
  const secret = new Set(
    specs.filter((spec) => spec.secret).map((spec) => spec.name),
  )
  return Object.fromEntries(
    Object.entries(definition?.create_kwargs ?? {}).filter(
      ([name]) => !secret.has(name),
    ),
  )
}

/**
 * The specs as the form should check and label them. A required secret the
 * definition already holds stops being required of the user, or every later
 * edit of another field would be refused over a box that is never prefilled.
 */
export function definitionFieldSpecs(
  specs: GuardrailParameterSpec[],
  held: ReadonlySet<string>,
): GuardrailParameterSpec[] {
  return specs.map((spec) =>
    spec.secret && held.has(spec.name)
      ? {
          ...spec,
          required: false,
          description: "Stored. Leave it blank to keep it, or type a new one.",
        }
      : spec,
  )
}

/**
 * The `create_kwargs` a save sends, whole.
 *
 * A held secret left blank is sent as `***`, which keeps the stored value.
 * **Do not leave it out instead**: the gateway reads an absent secret as
 * "clear it", so a save of the endpoint alone would silently delete the API
 * key. A blank secret the definition does not hold is left out, since there is
 * nothing to keep.
 *
 * Always a map, never `null`: `{}` is how a write says "no arguments".
 */
export function buildCreateKwargs(
  specs: GuardrailParameterSpec[],
  values: ParameterValues,
  held: ReadonlySet<string>,
): Record<string, unknown> {
  const kwargs = buildValidateKwargs(specs, values, "") ?? {}
  for (const spec of specs) {
    if (!spec.secret) continue
    const typed = values[spec.name]
    if (typeof typed === "string" && typed !== "") continue
    if (held.has(spec.name)) kwargs[spec.name] = REDACTED_SECRET
    else delete kwargs[spec.name]
  }
  return kwargs
}
