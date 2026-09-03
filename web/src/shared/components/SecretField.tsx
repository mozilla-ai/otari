import { Description, Input, Label, TextField } from "@heroui/react"
import { FieldMessages } from "@/shared/components/FieldMessages"

// A masked, never-prefilled secret input. Native password masking protects
// Firefox users; self-hosted deployments should use HTTPS to avoid browser
// warnings.
//
// Shared rather than page-local because the dashboard now collects two kinds of
// write-only credential: a provider API key, and an MCP server's bearer token.
// Both are stored encrypted and never read back, so a plain `Field` would show
// the one value on the form whose whole design is that it is never shown, and
// would offer it to a password manager as a login besides. The suppression
// attributes below are what stop that, and they are the reason this is one
// component rather than a `type="password"` flag on `Field`.
export function SecretField({
  value,
  onChange,
  label,
  placeholder,
  description,
  reserveMessage,
}: {
  value: string
  onChange: (next: string) => void
  label: string
  placeholder?: string
  description?: string
  /** See `Field`: holds one caption line open so a message does not move the
      form. Off for a field in a table row or a toolbar. */
  reserveMessage?: boolean
}) {
  return (
    <TextField
      value={value}
      onChange={onChange}
      className="flex max-w-md flex-col gap-1"
    >
      <Label className="text-body">{label}</Label>
      <Input
        type="password"
        placeholder={placeholder ?? "sk-…"}
        autoComplete="off"
        autoCorrect="off"
        autoCapitalize="off"
        spellCheck={false}
        data-1p-ignore
        data-lpignore="true"
      />
      <FieldMessages reserve={reserveMessage}>
        {description ? (
          <Description className="text-muted">{description}</Description>
        ) : null}
      </FieldMessages>
    </TextField>
  )
}
