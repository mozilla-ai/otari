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

import { useProviderCatalog } from "@/shared/api/hooks"
import { FieldMessages } from "@/shared/components/FieldMessages"

// The two form controls a provider credential needs wherever it is edited, and
// the parsing that goes with one of them.
//
// Shared because the same credential is entered on two pages that are otherwise
// unrelated: `/providers`, where it belongs to the process, and
// `/organization/provider-keys`, where it belongs to the tenant. Both take a
// provider name any-llm has to recognize and both take `client_args`, so a
// second copy of either control would be a second place for the JSON guard and
// the catalog lookup to drift.

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

// The client_args editor. Options are passed straight to the provider client, so
// a bad value is rejected here rather than sent (issue #517).
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
      <Label className="text-sm font-medium text-foreground">
        Client options (JSON)
      </Label>
      <TextArea
        rows={3}
        placeholder={'{"timeout": 1800}'}
        spellCheck={false}
        className="font-mono text-xs"
      />
      <FieldMessages>
        <Description
          className={
            error ? "text-caption text-danger" : "text-caption text-muted"
          }
        >
          {error ??
            // Unlike the API key, these are stored and returned unencrypted, so say
            // so before someone puts a token in a custom header here.
            "Passed to the provider's client, e.g. a request timeout in seconds or custom headers. Stored in plain text, so keep secrets out."}
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
}) {
  const catalog = useProviderCatalog()
  const options = useMemo(
    () =>
      includeCatalog
        ? [
            ...extra,
            ...(catalog.data ?? []).map((p) => ({ id: p.id, name: p.name })),
          ]
        : extra,
    [catalog.data, extra, includeCatalog],
  )

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
      <Label className="text-sm font-medium text-foreground">{label}</Label>
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
      {description ? (
        <span className="text-xs text-muted">{description}</span>
      ) : null}
    </ComboBox.Root>
  )
}
