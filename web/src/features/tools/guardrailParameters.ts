/**
 * Turning a guardrail profile's parameter schema into form state, and back into
 * the `validate_kwargs` dict the entry stores.
 *
 * A stored parameter whose name looks credential-shaped reads back as
 * `REDACTED_SECRET` rather than its value, and `validate_kwargs` is sent whole
 * on every save, so the mask has to travel back out untouched: the gateway
 * restores the stored value wherever it sees one, and an entry that arrived
 * masked and left blank would delete a credential on a save of something else.
 * That is why the three places a value is read, checked or coerced below each
 * let the mask through ahead of anything the schema says about its type.
 *
 * The schema comes from `GET /v1/tool-settings/guardrails/profiles`, which joins
 * the operator's own profile list to the `any_guardrail` parameter registry. So
 * a field here exists because a guardrail actually takes it, and a profile the
 * catalog cannot describe still round-trips: whatever the schema does not name
 * stays in the raw editor and is written back untouched.
 *
 * Every value is held as the string the operator typed, except a boolean, and is
 * coerced back to its JSON-native shape on submit. Holding a half-typed number
 * as a number means holding `NaN` while someone types a minus sign.
 */

import type { GuardrailCatalog, GuardrailParameterSpec } from "@/client"
import { REDACTED_SECRET } from "@/shared/helpers/redaction"

export type ParameterValue = string | boolean
export type ParameterValues = Record<string, ParameterValue>
export type ParameterErrors = Record<string, string | undefined>

/** A snake_case parameter name as a sentence. */
export function parameterLabel(name: string): string {
  const words = name.split("_").filter(Boolean)
  if (words.length === 0) return name
  return words
    .map((word, index) =>
      index === 0 ? word.charAt(0).toUpperCase() + word.slice(1) : word,
    )
    .join(" ")
}

/** The catalog entry for a profile, or undefined when it names none. */
export function findProfile(
  catalog: GuardrailCatalog | undefined,
  profile: string,
) {
  return catalog?.profiles?.find((entry) => entry.profile === profile)
}

/** The parameters of a profile the catalog describes, and none otherwise. */
export function parameterSpecs(
  catalog: GuardrailCatalog | undefined,
  profile: string,
): GuardrailParameterSpec[] {
  return findProfile(catalog, profile)?.parameters ?? []
}

/** Stands in for every profile name the catalog does not describe. */
const UNDESCRIBED_PROFILE = "\u0000undescribed"

/**
 * What a parameter form re-seeds on, beside the schema and the stored values.
 *
 * Two profiles of one guardrail class differing only in the model they pin
 * declare the same parameters, so a schema cannot tell them apart and a form
 * keyed on it alone carries one profile's values onto the other. A name can.
 *
 * Every name the catalog does not describe shares one identity, because such a
 * name is typed a character at a time: keying on it would reset the form on
 * every keystroke, and there is no schema behind it to re-seed from anyway.
 */
export function profileIdentity(
  catalog: GuardrailCatalog | undefined,
  profile: string,
): string {
  return findProfile(catalog, profile) === undefined
    ? UNDESCRIBED_PROFILE
    : profile
}

function isBlank(value: ParameterValue | undefined): boolean {
  return (
    value === undefined || (typeof value === "string" && value.trim() === "")
  )
}

function asFieldValue(
  spec: GuardrailParameterSpec,
  stored: unknown,
): ParameterValue {
  // Ahead of the boolean branch, which would otherwise read a masked value as
  // `false` and save that over the stored one.
  if (stored === REDACTED_SECRET) return REDACTED_SECRET
  if (spec.type === "boolean") return stored === true
  if (stored === null || stored === undefined) return ""
  if (typeof stored === "object") return JSON.stringify(stored, null, 2)
  return String(stored)
}

function defaultFieldValue(_spec: GuardrailParameterSpec): ParameterValue {
  // A default is shown only as the placeholder, never prefilled: writing it into
  // the field would store an explicit value where the profile's own default was
  // meant to apply, and the two stop tracking each other the moment the operator
  // upgrades the guardrails service.
  //
  // A boolean is blank here for the same reason and not because a checkbox has
  // three states: it has two, and blank is the third thing the *stored entry*
  // has, which is no opinion at all. Seeding an unset optional boolean as
  // `false` made a stored `false` indistinguishable from an absent key, so the
  // next save of the row dropped the operator's own "off" and handed the
  // profile's default back.
  return ""
}

export type SeededParameters = {
  values: ParameterValues
  /** Stored keys the schema does not name, kept verbatim for the raw editor. */
  extraJson: string
}

/**
 * Seed the form from an entry's stored `validate_kwargs`.
 *
 * A key the schema knows becomes a typed field; anything else goes to the raw
 * editor, so a profile configured before the catalog existed (or one the catalog
 * cannot describe) keeps every value it had.
 */
export function seedParameters(
  specs: GuardrailParameterSpec[],
  stored: Record<string, unknown> | null | undefined,
): SeededParameters {
  const kwargs = stored ?? {}
  const values: ParameterValues = {}
  const known = new Set<string>()
  for (const spec of specs) {
    known.add(spec.name)
    values[spec.name] = Object.hasOwn(kwargs, spec.name)
      ? asFieldValue(spec, kwargs[spec.name])
      : defaultFieldValue(spec)
  }
  const extra = Object.fromEntries(
    Object.entries(kwargs).filter(([key]) => !known.has(key)),
  )
  return {
    values,
    extraJson:
      Object.keys(extra).length > 0 ? JSON.stringify(extra, null, 2) : "",
  }
}

/** Parse the raw editor's contents, which must be a JSON object or nothing. */
export function parseExtraJson(raw: string): {
  value?: Record<string, unknown>
  error?: string
} {
  const text = raw.trim()
  if (text === "") return { value: {} }
  let parsed: unknown
  try {
    parsed = JSON.parse(text)
  } catch {
    return { error: "Not valid JSON." }
  }
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    return { error: 'Must be a JSON object, like {"threshold": 0.8}.' }
  }
  return { value: parsed as Record<string, unknown> }
}

function parseJsonValue(raw: string): { value?: unknown; error?: string } {
  try {
    return { value: JSON.parse(raw) as unknown }
  } catch {
    return { error: "Not valid JSON." }
  }
}

/**
 * Per-field messages, keyed by parameter name. Empty when the form is submittable.
 *
 * Run on submit rather than per keystroke, which is what the forms guide asks
 * for: a number field complaining at the minus sign is telling the operator they
 * are wrong before they have finished being right.
 */
export function parameterErrors(
  specs: GuardrailParameterSpec[],
  values: ParameterValues,
): ParameterErrors {
  const errors: ParameterErrors = {}
  for (const spec of specs) {
    const value = values[spec.name]
    if (spec.type === "boolean") continue
    // The mask stands in for a value nobody here has seen, so there is nothing
    // to check: a secret parameter the schema types as a number or JSON would
    // otherwise fail on the literal `***` and refuse a save of the rest of the
    // entry.
    if (value === REDACTED_SECRET) continue
    if (isBlank(value)) {
      if (spec.required)
        errors[spec.name] = "This guardrail needs a value here."
      continue
    }
    const text = String(value)
    if (spec.type === "integer" && !Number.isInteger(Number(text))) {
      errors[spec.name] = "Enter a whole number."
    } else if (spec.type === "number" && !Number.isFinite(Number(text))) {
      errors[spec.name] = "Enter a number."
    } else if (
      spec.type === "enum" &&
      spec.choices &&
      !spec.choices.includes(text)
    ) {
      errors[spec.name] = "Choose one of the listed values."
    } else if (spec.type === "json" && parseJsonValue(text).error) {
      errors[spec.name] = "Not valid JSON."
    }
  }
  return errors
}

function coerce(
  spec: GuardrailParameterSpec,
  value: ParameterValue | undefined,
): unknown {
  // Sent verbatim so the gateway can put the stored value back; coercing it
  // would send `NaN` for a numeric secret and `undefined` for a JSON one.
  if (value === REDACTED_SECRET) return REDACTED_SECRET
  if (spec.type === "boolean") return value === true
  const text = String(value)
  // `Number` for both, which is the parser `parameterErrors` approved the input
  // with. `parseInt` and `parseFloat` read a prefix instead of the whole string,
  // so "1e3" validates as the integer 1000 and would then be sent as 1.
  if (spec.type === "integer" || spec.type === "number") return Number(text)
  if (spec.type === "json") return parseJsonValue(text).value
  return text
}

/**
 * Assemble the typed fields and the raw editor into one `validate_kwargs`.
 *
 * The raw editor is the base and typed fields land on top, so a key in both is
 * the field's: it is the one the operator can see. A blank optional field is
 * omitted rather than sent empty, which leaves the profile's own default in
 * force on the guardrails service.
 *
 * Returns `null` for an entry that configures nothing, which is what the API
 * stores as "no kwargs": an empty object would read as a decision.
 */
export function buildValidateKwargs(
  specs: GuardrailParameterSpec[],
  values: ParameterValues,
  extraJson: string,
): Record<string, unknown> | null {
  const kwargs: Record<string, unknown> = {
    ...(parseExtraJson(extraJson).value ?? {}),
  }
  for (const spec of specs) {
    const value = values[spec.name]
    // Blank is "no opinion" for every type, a boolean included: an optional box
    // nobody has touched leaves the profile's own default in force, while one
    // the operator turned off sends `false` and keeps it off. A required
    // parameter has no such state, so its blank still sends.
    if (isBlank(value) && !spec.required) continue
    kwargs[spec.name] = coerce(spec, value)
  }
  return Object.keys(kwargs).length > 0 ? kwargs : null
}
