import type { ReactNode } from "react"
import {
  CheckboxGroup as AriaCheckboxGroup,
  CheckboxButton,
  CheckboxField,
  Label,
  Text,
} from "react-aria-components"

import { CheckboxVisual } from "./Checkbox"
import { FieldMessages } from "./FieldMessages"

/** One choice in a `CheckboxGroup`. `description` explains a choice whose label cannot. */
export interface CheckboxOption {
  value: string
  label: string
  description?: string
  isDisabled?: boolean
}

/**
 * Several of a short set, with every option on screen at once. When to reach
 * for it rather than `MultiSelect` is in design/forms.md.
 *
 * react-aria rather than HeroUI's own `CheckboxGroup`, for the reason `Checkbox`
 * gives: HeroUI splits the control across subcomponents and the two would not
 * look alike. `CheckboxVisual` is imported rather than redrawn, so a box here, a
 * standalone `Checkbox` and a table's selection box cannot drift apart.
 */
export function CheckboxGroup({
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
  value: readonly string[]
  onChange: (value: string[]) => void
  options: readonly CheckboxOption[]
  description?: ReactNode
  /** `horizontal` for short labels that fit a row; it wraps rather than scrolls. */
  orientation?: "vertical" | "horizontal"
  isRequired?: boolean
  isDisabled?: boolean
  isInvalid?: boolean
  errorMessage?: string
  className?: string
}) {
  return (
    <AriaCheckboxGroup
      // A copy rather than the array itself: react-aria types the value as a
      // mutable `string[]`, and handing it the caller's `readonly` one would
      // make every call site widen a prop it is right to keep narrow.
      value={[...value]}
      onChange={onChange}
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
          <CheckboxField
            key={option.value}
            value={option.value}
            isDisabled={option.isDisabled}
            className="flex w-fit flex-col"
          >
            <CheckboxButton
              // `items-center`, where `RadioGroup` starts its options: that one
              // bakes a 2px nudge into its own indicator, and `CheckboxVisual`
              // carries none because `Checkbox` centers it. `min-h-11` is the
              // 44px touch floor, which also gives the box's enlarged hit area a
              // row to fit inside rather than overlapping the next option.
              className="group flex min-h-11 items-center gap-2 text-body"
            >
              {({ isSelected, isDisabled: optionDisabled }) => (
                <>
                  <CheckboxVisual
                    isSelected={isSelected}
                    isIndeterminate={false}
                    isDisabled={optionDisabled}
                  />
                  {option.label}
                </>
              )}
            </CheckboxButton>
            {option.description ? (
              // Outside the button so it describes the option rather than
              // joining its name; indented past the box and its gap to sit
              // under the label.
              <Text slot="description" className="pl-6 text-caption">
                {option.description}
              </Text>
            ) : null}
          </CheckboxField>
        ))}
      </div>
      <FieldMessages shouldReserve={false}>
        {isInvalid && errorMessage ? (
          <Text slot="errorMessage" className="text-danger">
            {errorMessage}
          </Text>
        ) : null}
      </FieldMessages>
    </AriaCheckboxGroup>
  )
}
