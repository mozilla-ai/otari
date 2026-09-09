import {
  Description,
  FieldError,
  TextArea as HeroTextArea,
  Label,
  TextField,
} from "@heroui/react"
import type { ReactNode } from "react"

import { FieldMessages } from "./FieldMessages"

/**
 * Free text over more than one line.
 *
 * `Field`'s sibling, sharing its whole rig: a visible associated label, one
 * caption line under it carrying either the description or the error, and the
 * same optional reserve so a message does not move the form.
 *
 * The one place it deliberately differs from every other control here is its
 * height. A textarea keeps a *floor* rather than a fixed height, because it is
 * the one field that should grow with what somebody typed; `rows` sets where it
 * starts, and the browser's own resize handle takes it from there.
 */
export function TextArea({
  label,
  value,
  onChange,
  placeholder,
  rows = 4,
  description,
  isRequired,
  isDisabled,
  isInvalid,
  errorMessage,
  reserveMessage,
  className = "",
}: {
  label: string
  value: string
  onChange: (value: string) => void
  placeholder?: string
  /** Where the field starts. It still grows past this. */
  rows?: number
  description?: ReactNode
  isRequired?: boolean
  isDisabled?: boolean
  isInvalid?: boolean
  errorMessage?: string
  reserveMessage?: boolean
  className?: string
}) {
  return (
    <TextField
      value={value}
      onChange={onChange}
      isRequired={isRequired}
      isDisabled={isDisabled}
      isInvalid={isInvalid}
      className={`flex flex-col gap-1 ${className}`}
    >
      <Label className="text-body">{label}</Label>
      <HeroTextArea rows={rows} placeholder={placeholder} />
      <FieldMessages reserve={reserveMessage}>
        {description ? (
          <Description className="text-muted">{description}</Description>
        ) : null}
        {errorMessage ? (
          <FieldError className="text-danger">{errorMessage}</FieldError>
        ) : null}
      </FieldMessages>
    </TextField>
  )
}
