import type { SearchProviderInfo, SearchProviderOption } from "@/client"
import { Field } from "@/design-system/forms/Field"
import { Select } from "@/design-system/forms/Select"
import { TextArea } from "@/design-system/forms/TextArea"

import { jsonOnlyOptions, typedOptions } from "./webTools"

// A provider's options, from the catalog's schema: a field for each option the
// form can type, and a JSON box for the rest. The box is never optional: an
// object or array option has no field, a provider with no schema (`searxng`)
// has none at all, and either way the tool must still be configurable.

/** What the provider does when the option is not set, for the "Default" choice. */
function defaultLabel(option: SearchProviderOption): string {
  if (option.default === undefined || option.default === null) return "Default"
  if (typeof option.default === "boolean") {
    return `Default (${option.default ? "on" : "off"})`
  }
  return `Default (${String(option.default)})`
}

function OptionField({
  option,
  value,
  error,
  onChange,
}: {
  option: SearchProviderOption
  value: string
  error: string
  onChange: (value: string) => void
}) {
  const fallback = defaultLabel(option)
  if (option.type === "boolean" || (option.enum && option.enum.length > 0)) {
    const choices =
      option.type === "boolean"
        ? [
            { value: "true", label: "On" },
            { value: "false", label: "Off" },
          ]
        : (option.enum ?? []).map((choice) => ({
            value: choice,
            label: choice,
          }))
    return (
      <Select
        label={option.name}
        value={value}
        onChange={onChange}
        placeholder={fallback}
        options={[{ value: "", label: fallback }, ...choices]}
        description={option.description || undefined}
        isInvalid={error !== ""}
        errorMessage={error || undefined}
      />
    )
  }
  const isNumeric = option.type === "integer" || option.type === "number"
  return (
    <Field
      label={option.name}
      value={value}
      onChange={onChange}
      inputMode={isNumeric ? "decimal" : "text"}
      placeholder={
        option.default === undefined || option.default === null
          ? ""
          : String(option.default)
      }
      description={option.description || undefined}
      isInvalid={error !== ""}
      errorMessage={error || undefined}
    />
  )
}

export function ProviderOptionFields({
  provider,
  values,
  errors,
  extraJson,
  jsonError,
  onChange,
  onExtraJsonChange,
}: {
  provider: SearchProviderInfo | undefined
  /** Each typed option's draft, `""` for unset. */
  values: Record<string, string>
  errors: Record<string, string>
  extraJson: string
  jsonError: string
  onChange: (values: Record<string, string>) => void
  onExtraJsonChange: (text: string) => void
}) {
  const fields = typedOptions(provider)
  const jsonOnly = jsonOnlyOptions(provider)
  const jsonHelp =
    provider?.options === null || provider?.options === undefined
      ? "This provider publishes no option schema, so its options are sent as written."
      : jsonOnly.length > 0
        ? `For ${jsonOnly.join(", ")}, and any option above. The fields above win over the same key here.`
        : "Any option above, as JSON. The fields above win over the same key here."

  return (
    <div className="flex flex-col gap-4">
      {fields.map((option) => (
        <OptionField
          key={option.name}
          option={option}
          value={values[option.name] ?? ""}
          error={errors[option.name] ?? ""}
          onChange={(value) => onChange({ ...values, [option.name]: value })}
        />
      ))}
      <TextArea
        label="Other options (JSON)"
        value={extraJson}
        onChange={onExtraJsonChange}
        rows={3}
        placeholder={
          jsonOnly.length > 0 ? `{"${jsonOnly[0]}": …}` : '{"option": "value"}'
        }
        description={jsonHelp}
        isInvalid={jsonError !== ""}
        errorMessage={jsonError || undefined}
      />
    </div>
  )
}
