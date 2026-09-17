import type { ReactNode } from "react"

import type { GuardrailParameterSpec } from "@/client"
import { Checkbox } from "@/design-system/forms/Checkbox"
import { Field } from "@/design-system/forms/Field"
import { FieldMessages } from "@/design-system/forms/FieldMessages"
import { SecretField } from "@/design-system/forms/SecretField"
import { FilterSelect } from "@/design-system/navigation/FilterSelect"
import { DocstringText } from "@/features/tools/DocstringText"
import { GuardrailJsonField } from "@/features/tools/GuardrailJsonField"
import {
  type ParameterErrors,
  type ParameterValues,
  parameterLabel,
} from "@/features/tools/guardrailParameters"

// One control per parameter the chosen profile accepts, picked by the type the
// catalog reports. The whole point of the catalog is that this file names no
// guardrail and no parameter: it renders whatever the guardrails service says
// its profiles take.

/** What a field shows when it is left blank, which is the profile's own default. */
function placeholderFor(spec: GuardrailParameterSpec): string | undefined {
  if (spec.default === null || spec.default === undefined) return undefined
  const shown =
    typeof spec.default === "object"
      ? JSON.stringify(spec.default)
      : String(spec.default)
  return `default: ${shown}`
}

/**
 * The environment variable that fills a parameter left blank.
 *
 * Kept although the sentence carrying it is trimmed away: "or set $X" is the
 * difference between a required-looking field and one the deployment may
 * already answer, and it is three words rather than the clause upstream spends
 * on it.
 */
function envNote(spec: GuardrailParameterSpec): ReactNode {
  if (!spec.env_var) return null
  return (
    <>
      {" "}
      Or set <code className="text-mono-caption">{spec.env_var}</code>.
    </>
  )
}

/**
 * What a secret's field says about the value behind it.
 *
 * `SecretField` is never prefilled, so on an edit the only way to tell a stored
 * credential from an absent one is to say so. `storedSecrets` is undefined while
 * creating, where there is nothing to report and the schema's own help is what
 * the operator needs.
 */
function secretNote(
  spec: GuardrailParameterSpec,
  storedSecrets: readonly string[] | undefined,
): ReactNode {
  const help = (
    <>
      <DocstringText>{spec.description ?? undefined}</DocstringText>
      {envNote(spec)}
    </>
  )
  if (!spec.secret || storedSecrets === undefined) return help
  const status = storedSecrets.includes(spec.name)
    ? "Set already, and never shown again. Leave blank to keep it."
    : "Not set."
  return (
    <>
      {status} {help}
    </>
  )
}

function ParameterControl({
  spec,
  guardrailName,
  operation,
  scopeName,
  value,
  error,
  disabled,
  storedSecrets,
  onChange,
}: {
  spec: GuardrailParameterSpec
  /** The class the parameter belongs to, which is how a JSON suggestion is found. */
  guardrailName: string
  /** The category it was picked under. See `GuardrailJsonField`. */
  operation?: string
  /** Names the entry, so a parameter that repeats down the card is still distinct. */
  scopeName: string
  value: ParameterValues[string]
  error: string | undefined
  disabled: boolean
  /** See `secretNote`. Undefined for a form that is creating rather than editing. */
  storedSecrets?: readonly string[]
  onChange: (next: ParameterValues[string]) => void
}) {
  const label = parameterLabel(spec.name)
  // An unstorable argument is rendered rather than hidden, disabled and saying
  // why: upstream types it as a live client or session object, so no database
  // can hold one and the gateway refuses it. An absent field would leave the
  // operator hunting for the credential argument it is not.
  if (spec.storable === false) {
    return (
      <Field
        label={label}
        value=""
        onChange={() => undefined}
        isDisabled
        description="Cannot be stored: this argument takes a live client object. Use the credential arguments beside it."
        reserveMessage
      />
    )
  }
  const description: ReactNode = spec.secret ? (
    secretNote(spec, storedSecrets)
  ) : (
    <>
      <DocstringText>{spec.description ?? undefined}</DocstringText>
      {envNote(spec)}
    </>
  )

  if (spec.type === "boolean") {
    return (
      <div className="flex flex-col gap-1">
        <Checkbox
          isSelected={value === true}
          isDisabled={disabled}
          ariaLabel={`${label} for ${scopeName}`}
          onChange={onChange}
        >
          {label}
        </Checkbox>
        {description ? (
          <span className="text-caption">{description}</span>
        ) : null}
      </div>
    )
  }

  if (spec.secret) {
    // Masked whatever else the schema says it is: a credential must not be
    // echoed back into a visible box, and no coercion here needs to see it.
    return (
      <SecretField
        label={label}
        value={String(value ?? "")}
        onChange={onChange}
        placeholder="value"
        description={description}
        isDisabled={disabled}
        isInvalid={Boolean(error)}
        errorMessage={error}
        reserveMessage
      />
    )
  }

  if (spec.type === "enum" && spec.choices) {
    return (
      <div className="flex flex-col gap-1">
        <FilterSelect
          label={label}
          value={String(value ?? "")}
          disabled={disabled}
          onChange={onChange}
          // A blank option is what leaves the profile's own default in force,
          // so an optional enum stays clearable; a required one has no such
          // state and its blank entry is the unset the error message names.
          options={[
            { value: "", label: spec.required ? "Choose a value" : "Default" },
            ...spec.choices.map((choice) => ({ value: choice, label: choice })),
          ]}
        />
        <FieldMessages reserve>
          {error ? (
            <span className="text-caption text-danger">{error}</span>
          ) : description ? (
            <span className="text-caption">{description}</span>
          ) : null}
        </FieldMessages>
      </div>
    )
  }

  if (spec.type === "json") {
    return (
      <GuardrailJsonField
        spec={spec}
        guardrailName={guardrailName}
        operation={operation}
        value={String(value ?? "")}
        error={error}
        disabled={disabled}
        description={description}
        onChange={onChange}
      />
    )
  }

  return (
    <Field
      label={label}
      value={String(value ?? "")}
      onChange={onChange}
      isDisabled={disabled}
      isInvalid={Boolean(error)}
      errorMessage={error}
      placeholder={placeholderFor(spec)}
      description={description}
      reserveMessage
    />
  )
}

// No `isRequired` on these, deliberately. A required parameter is checked by
// `parameterErrors`, which names the field and says what it needs; the native
// attribute would refuse the submit first and silently, so inside a dialog the
// form would never reach that message. One path, and it is the one that speaks.
// The cost is the `required` marker these inputs no longer carry; the refusal
// still arrives on the field, announced, through `isInvalid` and `errorMessage`.
export function GuardrailParameterFields({
  specs,
  guardrailName,
  operation,
  scopeName,
  values,
  errors,
  disabled,
  storedSecrets,
  onChange,
}: {
  specs: GuardrailParameterSpec[]
  /** See `ParameterControl`. */
  guardrailName: string
  operation?: string
  scopeName: string
  values: ParameterValues
  errors: ParameterErrors
  disabled: boolean
  /** Secret names the entry already holds, which marks this an edit. See `secretNote`. */
  storedSecrets?: readonly string[]
  onChange: (name: string, next: ParameterValues[string]) => void
}) {
  return (
    <div className="flex flex-col gap-4 sm:grid sm:grid-cols-2 sm:items-start">
      {specs.map((spec) => (
        // A JSON argument brings a layout of its own, a column of switches or
        // of rows, and is several times the height of the one-line fields
        // beside it. Half a row leaves it cramped and its neighbour stranded
        // above a gap, so it takes the whole row and the flat fields pair up
        // around it.
        <div
          key={spec.name}
          className={
            spec.type === "json" && spec.storable !== false
              ? "sm:col-span-2"
              : undefined
          }
        >
          <ParameterControl
            spec={spec}
            guardrailName={guardrailName}
            operation={operation}
            scopeName={scopeName}
            value={values[spec.name]}
            error={errors[spec.name]}
            disabled={disabled}
            storedSecrets={storedSecrets}
            onChange={(next) => onChange(spec.name, next)}
          />
        </div>
      ))}
    </div>
  )
}
