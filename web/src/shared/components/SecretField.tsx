import { Description, Input, Label, TextField } from "@heroui/react"

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
}: {
  value: string
  onChange: (next: string) => void
  label: string
  placeholder?: string
  description?: string
}) {
  return (
    <TextField
      value={value}
      onChange={onChange}
      className="flex max-w-md flex-col gap-1"
    >
      <Label className="text-sm font-medium text-foreground">{label}</Label>
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
      {description ? (
        <Description className="text-xs text-muted">{description}</Description>
      ) : null}
    </TextField>
  )
}
