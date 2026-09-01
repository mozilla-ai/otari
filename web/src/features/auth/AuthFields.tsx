import { Description, Input, Label, TextField } from "@heroui/react"

/**
 * The three fields the pages in front of a session are built from.
 *
 * `shared/components/Field` is the general one and does not fit here: it caps
 * itself at `max-w-md` for a settings page and offers no `type="password"` or
 * `autoComplete`, both of which a credential form needs for a password manager
 * to file what it sees. The cards these render into are already narrow, so the
 * fields fill them.
 */

export function AuthEmailField({
  label = "Email",
  value,
  onChange,
  description,
  isReadOnly = false,
}: {
  label?: string
  value: string
  onChange: (next: string) => void
  description?: string
  /**
   * Shows the address without letting it be edited, for a form whose address
   * was decided elsewhere; `SignupPage` is the caller and says why. Still a
   * field rather than a line of text, so a password manager files the
   * credential it is being set beside against a username.
   */
  isReadOnly?: boolean
}) {
  return (
    <TextField
      value={value}
      onChange={onChange}
      type="email"
      isRequired
      isReadOnly={isReadOnly}
      className="flex flex-col gap-1"
    >
      <Label className="text-sm font-medium text-foreground">{label}</Label>
      {/* autoComplete="username" and not "email": this is the handle the
          sign-in form asks for, so a password manager should file it against
          the credential it is being set beside. */}
      {/* No autoFocus. These are pages, not dialogs, and focusing a field on
          mount raises the soft keyboard over the explanation above it before
          the visitor has asked to type (frontend-standards/responsiveness.md,
          and the same call `features/account/PasswordCard` makes). */}
      {/* A rule against the rendered input rather than a token or a HeroUI
          prop, which is the order the house style asks for and neither of
          which reaches this: HeroUI styles `isReadOnly` identically to an
          editable field, so without it the one field on the page that ignores
          typing looks exactly like the ones that do not. `bg-surface-alt` is
          the registered utility for `--color-surface-muted`; `bg-surface-muted`
          is declared nowhere and compiles to nothing (see `Login`'s CODE_CHIP). */}
      <Input
        placeholder="you@example.com"
        autoComplete="username"
        className="read-only:bg-surface-alt read-only:text-muted"
      />
      {description ? (
        // HeroUI's Description reaches the input as aria-describedby through
        // the TextField's "description" slot, which a raw span does not.
        <Description className="text-caption">{description}</Description>
      ) : null}
    </TextField>
  )
}

export function AuthPasswordField({
  label,
  value,
  onChange,
  autoComplete,
  description,
}: {
  label: string
  value: string
  onChange: (next: string) => void
  autoComplete: "current-password" | "new-password"
  description?: string
}) {
  return (
    <TextField
      value={value}
      onChange={onChange}
      type="password"
      isRequired
      className="flex flex-col gap-1"
    >
      <Label className="text-sm font-medium text-foreground">{label}</Label>
      <Input autoComplete={autoComplete} />
      {description ? (
        <Description className="text-caption">{description}</Description>
      ) : null}
    </TextField>
  )
}

export function AuthTextField({
  label,
  value,
  onChange,
  autoComplete,
  description,
}: {
  label: string
  value: string
  onChange: (next: string) => void
  autoComplete?: string
  description?: string
}) {
  return (
    <TextField
      value={value}
      onChange={onChange}
      className="flex flex-col gap-1"
    >
      <Label className="text-sm font-medium text-foreground">{label}</Label>
      <Input autoComplete={autoComplete} />
      {description ? (
        <Description className="text-caption">{description}</Description>
      ) : null}
    </TextField>
  )
}
