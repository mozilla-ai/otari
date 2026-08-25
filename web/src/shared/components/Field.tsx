import { Description, FieldError, Input, Label, TextField } from "@heroui/react"
import type { ReactNode } from "react"

interface FieldProps {
  label: string
  value: string
  onChange: (value: string) => void
  placeholder?: string
  type?: "text" | "datetime-local"
  isRequired?: boolean
  description?: ReactNode
  autoFocus?: boolean
  /** Marks the input invalid, which is what reddens its border. */
  isInvalid?: boolean
  /** Shown in the description's place while invalid. See the note below. */
  errorMessage?: ReactNode
}

// A labeled single-line text input built from HeroUI's TextField primitives.
export function Field({
  label,
  value,
  onChange,
  placeholder,
  type = "text",
  isRequired,
  description,
  autoFocus,
  isInvalid,
  errorMessage,
}: FieldProps) {
  // Either/or, never both: the two read on the same line under the input, so
  // showing them together would grow the field by a row. Swapping keeps whatever
  // frames the field (a modal, here) exactly the height it already was.
  const showsError = isInvalid === true && errorMessage !== undefined
  return (
    <TextField
      value={value}
      onChange={onChange}
      isRequired={isRequired}
      isInvalid={isInvalid}
      className="flex max-w-md flex-col gap-1"
    >
      {/* No manual "*": HeroUI marks a required field's label through CSS
          ([data-required=true] > .label::after), so adding one renders two. */}
      <Label className="text-sm font-medium text-foreground">{label}</Label>
      <Input type={type} placeholder={placeholder} autoFocus={autoFocus} />
      {showsError ? (
        // FieldError renders through the "error" slot, so it is announced as the
        // input's error rather than read as loose text next to it. HeroUI gives
        // it its own `text-danger` and collapses it to `h-0` until the field is
        // invalid, so nothing is reserved while the field is fine.
        <FieldError className="text-xs">{errorMessage}</FieldError>
      ) : description ? (
        // HeroUI's Description renders through the TextField's "description" slot,
        // so it is wired to the input via aria-describedby (a raw span is not).
        <Description className="text-xs text-muted">{description}</Description>
      ) : null}
    </TextField>
  )
}
