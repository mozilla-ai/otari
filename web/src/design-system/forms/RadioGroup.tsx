import {
  Description,
  RadioGroup as HeroRadioGroup,
  Label,
  Radio,
} from "@heroui/react"
import type { ReactNode } from "react"

import { FieldMessages } from "./FieldMessages"

/** One choice in a `RadioGroup`. `description` explains a choice whose label cannot. */
export interface RadioOption {
  value: string
  label: string
  description?: string
  isDisabled?: boolean
}

/**
 * One of a short set, with every option on screen at once.
 *
 * The line against `Select`: a radio group spends vertical space to make the
 * alternatives readable, so it is for a choice whose options need explaining
 * (three routing strategies, two retention policies) or where seeing all of
 * them at once is the point. Past about five options, or where the labels are
 * self-evident, `Select` is the smaller control.
 *
 * Not `Segmented`, either. That one is for a choice that filters what is on
 * screen right now, and it puts the options in a track shoulder to shoulder;
 * this is a form field the operator submits.
 */
export function RadioGroup({
  label,
  value,
  onChange,
  options,
  description,
  orientation = "vertical",
  isRequired,
  isDisabled,
  isInvalid,
  errorMessage,
  className = "",
}: {
  label: string
  value: string
  onChange: (value: string) => void
  options: readonly RadioOption[]
  description?: ReactNode
  /** `horizontal` only for two or three short labels; it wraps badly past that. */
  orientation?: "vertical" | "horizontal"
  isRequired?: boolean
  isDisabled?: boolean
  isInvalid?: boolean
  errorMessage?: string
  className?: string
}) {
  return (
    <HeroRadioGroup
      value={value}
      onChange={onChange}
      orientation={orientation}
      isRequired={isRequired}
      isDisabled={isDisabled}
      isInvalid={isInvalid}
      className={`flex flex-col gap-2 ${className}`}
    >
      <Label className="text-body">{label}</Label>
      {description ? (
        // Outside `FieldMessages` and with no reserve: this describes the whole
        // group and is never replaced by an error, so it is supporting text
        // rather than the message line the error competes for. The reserve
        // belongs to the line at the bottom, which is the one that changes.
        <Description className="text-caption">{description}</Description>
      ) : null}
      <div
        className={
          orientation === "horizontal"
            ? "flex flex-wrap items-center gap-4"
            : "flex flex-col gap-2"
        }
      >
        {options.map((option) => (
          <Radio
            key={option.value}
            value={option.value}
            isDisabled={option.isDisabled}
          >
            <span className="flex flex-col">
              <span className="text-body">{option.label}</span>
              {option.description ? (
                <span className="text-caption">{option.description}</span>
              ) : null}
            </span>
          </Radio>
        ))}
      </div>
      <FieldMessages reserve={false}>
        {isInvalid && errorMessage ? (
          <span className="text-danger">{errorMessage}</span>
        ) : null}
      </FieldMessages>
    </HeroRadioGroup>
  )
}
