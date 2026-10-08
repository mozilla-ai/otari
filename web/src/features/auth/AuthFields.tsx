import { Description, FieldError, Input, Label, TextField } from "@heroui/react"
import { useState } from "react"
import { FiEye, FiEyeOff } from "react-icons/fi"
import { Button } from "@/design-system/actions/Button"
import { FieldMessages } from "@/design-system/forms/FieldMessages"

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
  autoFocus = false,
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
  /**
   * Only for a field that mounts because the visitor asked for it (a folded
   * form they just opened). Never on a page load: that raises the soft keyboard
   * over the page before anyone has asked to type.
   */
  autoFocus?: boolean
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
      <Label className="text-body">{label}</Label>
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
        autoFocus={autoFocus}
        className="read-only:bg-surface-alt read-only:text-muted"
      />
      {description ? (
        <FieldMessages>
          {/* HeroUI's Description reaches the input as aria-describedby
              through the TextField's "description" slot, which a raw span
              does not. */}
          <Description className="text-muted">{description}</Description>
        </FieldMessages>
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
  errorMessage,
  canReveal = false,
}: {
  label: string
  value: string
  onChange: (next: string) => void
  autoComplete: "current-password" | "new-password"
  description?: string
  /**
   * Adds a toggle that shows what was typed. For a field whose value is chosen
   * rather than recalled, where a mistyped password is the likelier error and
   * there is no second field to catch it.
   */
  canReveal?: boolean
  /**
   * Why the password cannot be used yet, taking the description's line rather
   * than one of its own. The card these forms sit in is what the animated
   * background measures its bar grid from, so a message that changes the
   * card's height moves the whole field behind it (otari-ai#2146). Give this
   * to a field that carries a description, or it takes a line after all.
   */
  errorMessage?: string
}) {
  const [isRevealed, setIsRevealed] = useState(false)
  const input = (
    <Input
      autoComplete={autoComplete}
      className={canReveal ? "w-full pr-10" : undefined}
    />
  )
  return (
    <TextField
      value={value}
      onChange={onChange}
      type={canReveal && isRevealed ? "text" : "password"}
      isRequired
      isInvalid={Boolean(errorMessage)}
      className="flex flex-col gap-1"
    >
      <Label className="text-body">{label}</Label>
      {canReveal ? (
        <div className="relative">
          {input}
          {/* Centered on the field and inset 2px from its edge, so the glyph
              reads as inside the input. 32px to the eye at every width; the
              44px touch floor is the pseudo-element bleed, 7px from the padding
              box (the button's 1px transparent border takes one of them, so
              6px past its edge), which a 36px field can hold without the
              target overlapping anything. The `!` is the
              exception to the phone-width floor on `[data-slot="button"]`
              (globals.css), which would otherwise make the visible button 44
              and the bleed 56, into the label above. */}
          <Button
            type="button"
            variant="ghost"
            isIconOnly
            size="sm"
            aria-label={isRevealed ? "Hide password" : "Show password"}
            aria-pressed={isRevealed}
            onPress={() => setIsRevealed((shown) => !shown)}
            className="absolute inset-y-0 right-0.5 my-auto size-8 min-h-8! min-w-8! text-muted before:absolute before:-inset-[7px]"
          >
            {isRevealed ? (
              <FiEyeOff aria-hidden className="size-4" />
            ) : (
              <FiEye aria-hidden className="size-4" />
            )}
          </Button>
        </div>
      ) : (
        input
      )}
      {description || errorMessage ? (
        <FieldMessages>
          {/* `FieldError` renders through the field's error slot, so the
              message is announced on the input rather than sitting in the form
              as a loose paragraph; `isInvalid` above is what lets it render. */}
          {errorMessage ? (
            <FieldError className="text-danger">{errorMessage}</FieldError>
          ) : (
            <Description className="text-muted">{description}</Description>
          )}
        </FieldMessages>
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
      <Label className="text-body">{label}</Label>
      <Input autoComplete={autoComplete} />
      {description ? (
        <FieldMessages>
          <Description className="text-muted">{description}</Description>
        </FieldMessages>
      ) : null}
    </TextField>
  )
}
