import type { ReactNode } from "react"
import {
  Radio as AriaRadio,
  RadioGroup as AriaRadioGroup,
  Label,
  Text,
} from "react-aria-components"

import { FieldMessages } from "./FieldMessages"

/** One choice in a `RadioGroup`. `description` explains a choice whose label cannot. */
export interface RadioOption {
  value: string
  label: string
  description?: string
  isDisabled?: boolean
}

/**
 * The radio's indicator, square on the design tokens like `CheckboxVisual`.
 *
 * The whole product draws at `--radius: 0`, so this stays square and marks the
 * selected option with a filled dot where the checkbox marks it with a check.
 * The box fills with `--color-control-indicator` and the dot is
 * `--color-accent-glyph`, the same pairing the checkbox uses.
 */
export function RadioVisual({ isSelected }: { isSelected: boolean }) {
  return (
    <span
      className={`mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center transition-colors ${
        isSelected
          ? "bg-control-indicator text-accent-glyph"
          : "border border-control-border bg-background"
      } group-data-[focus-visible]:otari-focus-ring`}
    >
      {isSelected ? (
        <span className="h-2 w-2 bg-current" aria-hidden="true" />
      ) : null}
    </span>
  )
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
 *
 * react-aria rather than HeroUI's own `Radio`, for the reason `Checkbox` gives:
 * HeroUI splits the control across subcomponents, so its `Radio` root renders no
 * input and no indicator on its own.
 */
export function RadioGroup({
  label,
  hideLabel = false,
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
  /** Keep the label for assistive technology where the surrounding UI already shows it. */
  hideLabel?: boolean
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
    <AriaRadioGroup
      value={value}
      onChange={onChange}
      orientation={orientation}
      isRequired={isRequired}
      isDisabled={isDisabled}
      isInvalid={isInvalid}
      className={`flex flex-col gap-2 ${className}`}
    >
      <Label className={hideLabel ? "sr-only" : "text-body"}>{label}</Label>
      {description ? (
        // Supporting text for the whole group, so it sits outside the reserved
        // message line the error competes for at the bottom.
        <Text slot="description" className="text-caption">
          {description}
        </Text>
      ) : null}
      <div
        className={
          orientation === "horizontal"
            ? "flex flex-wrap items-center gap-4"
            : "flex flex-col gap-2"
        }
      >
        {options.map((option) => (
          <AriaRadio
            key={option.value}
            value={option.value}
            isDisabled={option.isDisabled}
            className="group flex w-fit items-start gap-2 text-body"
          >
            {({ isSelected }) => (
              <>
                <RadioVisual isSelected={isSelected} />
                <span className="flex flex-col">
                  <span className="text-body">{option.label}</span>
                  {option.description ? (
                    <span className="text-caption">{option.description}</span>
                  ) : null}
                </span>
              </>
            )}
          </AriaRadio>
        ))}
      </div>
      <FieldMessages shouldReserve={false}>
        {isInvalid && errorMessage ? (
          <span className="text-danger">{errorMessage}</span>
        ) : null}
      </FieldMessages>
    </AriaRadioGroup>
  )
}
