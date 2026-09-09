import {
  ComboBox,
  Description,
  Input,
  Label,
  ListBox,
  ListBoxItem,
  TextArea,
  TextField,
} from "@heroui/react"
import { type ReactNode, useMemo, useState } from "react"

import { useProviderCatalog } from "@/shared/api/providers"
import { Field } from "@/shared/components/forms/Field"
import { FieldMessages } from "@/shared/components/forms/FieldMessages"
import { SecretField } from "@/shared/components/forms/SecretField"

import {
  type CredentialFieldValues,
  credentialFieldsFor,
} from "./providerCredentialFields"

// The form controls a provider credential needs wherever it is edited, and the
// parsing that goes with them.
//
// Shared because the same credential is entered on two pages that are otherwise
// unrelated: `/providers`, where it belongs to the process, and
// `/organization/provider-keys`, where it belongs to the tenant. Both take a
// provider name any-llm has to recognize and both take `client_args`, so a
// second copy of any of these controls would be a second place for the JSON
// guard, the catalog lookup and the per-provider field list to drift.

// client_args is whatever the provider's SDK client constructor takes (timeouts,
// custom headers), so it has no fixed schema and the form edits it as JSON. Blank
// means "none": the API reads an explicit null as "clear it".
export type ClientArgsParse =
  | { ok: true; value: Record<string, unknown> | null }
  | { ok: false; error: string }

export function parseClientArgs(text: string): ClientArgsParse {
  const raw = text.trim()
  if (raw === "") return { ok: true, value: null }
  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch {
    return { ok: false, error: "Not valid JSON." }
  }
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    return {
      ok: false,
      error: 'Must be a JSON object, like {"timeout": 1800}.',
    }
  }
  return { ok: true, value: parsed as Record<string, unknown> }
}

// Render stored client_args back into the textarea, leaving it blank when there
// are none so an untouched form submits null rather than an empty object.
export function formatClientArgs(
  args: Record<string, unknown> | null | undefined,
): string {
  return args && Object.keys(args).length > 0
    ? JSON.stringify(args, null, 2)
    : ""
}

// The typed fields a provider expects inside `client_args`, from the registry.
// Renders nothing for the providers that need none, which is nearly all of them.
export function ProviderCredentialFields({
  provider,
  values,
  onChange,
  errors,
  redacted = [],
}: {
  provider: string
  values: CredentialFieldValues
  onChange: (next: CredentialFieldValues) => void
  /** Per-field messages from `validateCredentialFields`, keyed by field key. */
  errors: Record<string, string>
  /** Fields whose stored value came back masked, so blank means "keep it". */
  redacted?: readonly string[]
}) {
  const fields = credentialFieldsFor(provider)
  if (fields.length === 0) return null

  return (
    <>
      {fields.map((field) => {
        const value = values[field.key] ?? ""
        const error = errors[field.key]
        const set = (next: string) => onChange({ ...values, [field.key]: next })
        // The gateway masks anything credential-shaped by key name, so a stored
        // value is often unreadable whether or not the registry calls it a
        // secret. Say it is set instead of prefilling the mask.
        const description = redacted.includes(field.key)
          ? `Set already, and never shown again. Leave blank to keep it. ${field.helpText}`
          : field.helpText
        if (field.isSecret) {
          return (
            <SecretField
              key={field.key}
              label={field.label}
              value={value}
              onChange={set}
              placeholder={field.placeholder ?? "••••••••"}
              description={description}
              isInvalid={error !== undefined}
              errorMessage={error}
            />
          )
        }
        return (
          <Field
            key={field.key}
            label={field.label}
            value={value}
            onChange={set}
            isRequired={field.isRequired}
            placeholder={field.placeholder}
            description={description}
            isInvalid={error !== undefined}
            errorMessage={error}
          />
        )
      })}
    </>
  )
}

// The client_args editor: the escape hatch for whatever the typed fields above
// do not describe. Options are passed straight to the provider client, so a bad
// value is rejected here rather than sent (issue #517).
export function ClientArgsField({
  value,
  onChange,
  error,
}: {
  value: string
  onChange: (next: string) => void
  error: string | null
}) {
  return (
    <TextField
      value={value}
      onChange={onChange}
      isInvalid={error !== null}
      className="flex max-w-md flex-col gap-1"
    >
      <Label className="text-body">Client options (JSON)</Label>
      <TextArea
        rows={3}
        placeholder={'{"timeout": 1800}'}
        spellCheck={false}
        className="font-mono text-xs"
      />
      <FieldMessages>
        <Description
          className={error ? "text-caption text-danger" : "text-caption"}
        >
          {error ??
            // Both halves of that sentence are load-bearing, and blanket "keep
            // secrets out" advice would be wrong: Bedrock's classic IAM shape
            // genuinely needs a secret in here (`gateway/models/provider_keys.py`),
            // `redact_secret_like_values` is why it does not come back, and
            // `encrypted_api_key` is the protection it does not get.
            "Passed to the provider's client, e.g. a request timeout in seconds or custom headers. An option named like a credential is masked when read back, but nothing here is encrypted at rest."}
        </Description>
      </FieldMessages>
    </TextField>
  )
}

// A searchable provider picker over the known-provider catalog. Selection sets
// an id (provider id, or a provider_type) while the input shows the display
// name. `extra` prepends synthetic options like "OpenAI-compatible".
export function ProviderComboBox({
  label,
  value,
  onChange,
  description,
  placeholder,
  extra = [],
  includeCatalog = true,
  excludeIds,
}: {
  label: string
  value: string
  onChange: (id: string) => void
  description?: ReactNode
  placeholder?: string
  extra?: { id: string; name: string }[]
  // When false, offer only `extra` (e.g. the two API dialects), not the full
  // provider catalog.
  includeCatalog?: boolean
  // Catalog entries to leave out, for a form that cannot honor them. Per call
  // site rather than a rule of the picker: which providers are offerable
  // depends on what the form collects, not on the catalog. See
  // `BYO_UNSUPPORTED_PROVIDERS`.
  excludeIds?: readonly string[]
}) {
  const catalog = useProviderCatalog()
  const options = useMemo(() => {
    const catalogOptions = includeCatalog
      ? (catalog.data ?? [])
          .filter((p) => !excludeIds?.includes(p.id))
          .map((p) => ({ id: p.id, name: p.name }))
      : []
    return [...extra, ...catalogOptions]
  }, [catalog.data, extra, includeCatalog, excludeIds])

  // Seed the input with the selected option's display name. The field owns its
  // text after mount (updated on typing and on selection); syncing it back from
  // `value` on every render would wipe out what the user is typing, since the
  // options array is recreated each render.
  const [text, setText] = useState(
    () => options.find((o) => o.id === value)?.name ?? "",
  )

  // When the input merely shows the current selection, treat the query as empty
  // so opening the dropdown reveals every option, not just the selected one.
  const selectedName = options.find((o) => o.id === value)?.name ?? ""
  const query =
    text.trim() === selectedName.trim() ? "" : text.trim().toLowerCase()
  const visible = options
    .filter(
      (o) =>
        !query ||
        o.name.toLowerCase().includes(query) ||
        o.id.toLowerCase().includes(query),
    )
    .slice(0, 50)

  return (
    <ComboBox.Root
      allowsEmptyCollection
      // Open the full list on focus/click and filter as you type: this is a
      // pick-from-a-list control, not a free-text field, and it is not
      // autofocused, so the list does not spring open when the form appears.
      menuTrigger="focus"
      inputValue={text}
      onInputChange={setText}
      onSelectionChange={(key) => {
        if (key != null) {
          onChange(String(key))
          setText(options.find((o) => o.id === String(key))?.name ?? "")
        } else {
          // Selection cleared: clear the parent value too, so the submitted
          // data cannot keep a stale provider after the field is emptied.
          onChange("")
          setText("")
        }
      }}
      className="flex max-w-md flex-col gap-1"
    >
      <Label className="text-body">{label}</Label>
      <ComboBox.InputGroup>
        {/* Not a credential field: keep browser password managers from offering to fill it.
            Select the text on focus so typing replaces the current selection instead of
            appending to it (otherwise "OpenAI-compatible" + typing filters to nothing). */}
        <Input
          placeholder={placeholder ?? "Search providers…"}
          autoComplete="off"
          data-1p-ignore
          data-lpignore="true"
          onFocus={(event) => event.currentTarget.select()}
        />
        <ComboBox.Trigger />
      </ComboBox.InputGroup>
      <ComboBox.Popover>
        <ListBox items={visible} className="max-h-72 overflow-auto">
          {(option: { id: string; name: string }) => (
            <ListBoxItem id={option.id} textValue={option.name}>
              {option.name}
            </ListBoxItem>
          )}
        </ListBox>
      </ComboBox.Popover>
      {description ? <span className="text-caption">{description}</span> : null}
    </ComboBox.Root>
  )
}
