import { Description, FieldError, Input, Label, TextField } from "@heroui/react"
import type { ReactNode } from "react"

interface FieldProps {
  label: string
  value: string
  onChange: (value: string) => void
  placeholder?: string
  type?: "text" | "datetime-local"
  isRequired?: boolean
  /** Renders the input read-only and dimmed. A caller that cannot write should
      say so on the field rather than only on the button that commits it. */
  isDisabled?: boolean
  description?: ReactNode
  autoFocus?: boolean
  /** Marks the input invalid, which is what makes `errorMessage` render. */
  isInvalid?: boolean
  /** Shown under the field and announced with it. Needs `isInvalid` to appear. */
  errorMessage?: string
}

// A labeled single-line text input built from HeroUI's TextField primitives.
export function Field({
  label,
  value,
  onChange,
  placeholder,
  type = "text",
  isRequired,
  isDisabled,
  description,
  autoFocus,
  isInvalid,
  errorMessage,
}: FieldProps) {
  return (
    <TextField
      value={value}
      onChange={onChange}
      isRequired={isRequired}
      isDisabled={isDisabled}
      isInvalid={isInvalid}
      className="flex max-w-md flex-col gap-1"
    >
      {/* No manual "*": HeroUI marks a required field's label through CSS
          ([data-required=true] > .label::after), so adding one renders two. */}
      <Label className="text-sm font-medium text-foreground">{label}</Label>
      <Input type={type} placeholder={placeholder} autoFocus={autoFocus} />
      {description ? (
        // HeroUI's Description renders through the TextField's "description" slot,
        // so it is wired to the input via aria-describedby (a raw span is not).
        <Description className="text-xs text-muted">{description}</Description>
      ) : null}
      {/* Same reasoning one step further: `FieldError` renders through the
          field's error slot, so the message is announced *on* the input rather
          than sitting somewhere else in the form as a loose paragraph. It only
          renders while the field is invalid, which is why `isInvalid` gates it. */}
      {errorMessage ? (
        <FieldError className="text-xs text-danger">{errorMessage}</FieldError>
      ) : null}
    </TextField>
  )
}
