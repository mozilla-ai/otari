import { Description, Label, TextArea, TextField } from "@heroui/react"

import type { GuardrailParameterSpec } from "@/client"
import { Checkbox } from "@/design-system/forms/Checkbox"
import { Field } from "@/design-system/forms/Field"
import { FieldMessages } from "@/design-system/forms/FieldMessages"
import { SecretField } from "@/design-system/forms/SecretField"
import { FilterSelect } from "@/design-system/navigation/FilterSelect"
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

function ParameterControl({
  spec,
  scopeName,
  value,
  error,
  disabled,
  onChange,
}: {
  spec: GuardrailParameterSpec
  /** Names the entry, so a parameter that repeats down the card is still distinct. */
  scopeName: string
  value: ParameterValues[string]
  error: string | undefined
  disabled: boolean
  onChange: (next: ParameterValues[string]) => void
}) {
  const label = parameterLabel(spec.name)
  const description = spec.description ?? undefined

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
      <TextField
        value={String(value ?? "")}
        onChange={onChange}
        isDisabled={disabled}
        isInvalid={Boolean(error)}
        className="flex max-w-md flex-col gap-1"
      >
        <Label className="text-body">{label} (JSON)</Label>
        <TextArea
          rows={3}
          spellCheck={false}
          placeholder={placeholderFor(spec) ?? "[]"}
          className="font-mono text-xs"
        />
        <FieldMessages reserve>
          <Description
            className={error ? "text-caption text-danger" : "text-caption"}
          >
            {error ?? description}
          </Description>
        </FieldMessages>
      </TextField>
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
  scopeName,
  values,
  errors,
  disabled,
  onChange,
}: {
  specs: GuardrailParameterSpec[]
  scopeName: string
  values: ParameterValues
  errors: ParameterErrors
  disabled: boolean
  onChange: (name: string, next: ParameterValues[string]) => void
}) {
  return (
    <div className="flex flex-col gap-3 sm:grid sm:grid-cols-2">
      {specs.map((spec) => (
        <ParameterControl
          key={spec.name}
          spec={spec}
          scopeName={scopeName}
          value={values[spec.name]}
          error={errors[spec.name]}
          disabled={disabled}
          onChange={(next) => onChange(spec.name, next)}
        />
      ))}
    </div>
  )
}
