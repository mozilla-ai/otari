import {
  Description,
  Select as HeroSelect,
  Label,
  ListBox,
  ListBoxItem,
} from "@heroui/react"
import type { ReactNode } from "react"

import { FieldMessages } from "./FieldMessages"
import { optionKey, optionValue } from "./optionKey"

/**
 * One choice in a `Select`. `isDisabled` shows a choice that exists but cannot
 * be taken. `description` is one sentence under the label in the open list,
 * for a set whose labels alone do not say what each choice does; the closed
 * control still shows the label only.
 */
export interface SelectOption {
  value: string
  label: string
  description?: string
  isDisabled?: boolean
}

/**
 * One of a short, closed set, in a form the operator submits.
 *
 * The form counterpart to `navigation/FilterSelect`, and they are deliberately
 * two components rather than one with a `context` prop. A filter's label is a
 * caption beside the control and it never speaks a validation message; a form
 * field's label sits above it, is always visible, and owns a description and an
 * error line that has to be announced on the control. Those are different
 * components wearing the same trigger, and the prop list that served both would
 * be mostly ignored at each call site.
 *
 * On HeroUI's `Select` rather than a native `<select>` for the reason
 * `FilterSelect` gives: a native menu draws itself *over* the control on macOS,
 * covering the trigger that opened it.
 */
export function Select({
  label,
  value,
  onChange,
  options,
  description,
  labelAction,
  placeholder = "Select an option",
  autoFocus,
  isRequired,
  isDisabled,
  isInvalid,
  errorMessage,
  shouldReserveMessage,
  className = "",
}: {
  label: string
  value: string
  /** Takes the value, never an event, which is the convention every control here follows. */
  onChange: (value: string) => void
  options: readonly SelectOption[]
  description?: ReactNode
  /**
   * A small control on the label's line, such as an icon that explains the
   * field. A native element: a react-aria `Button` here would read the select's
   * own button context and become a second trigger.
   */
  labelAction?: ReactNode
  /** Shown while nothing is selected. An example, never the label. */
  placeholder?: string
  /** A form's first field takes this; see feedback.md. Same as `ComboBoxField`'s. */
  autoFocus?: boolean
  isRequired?: boolean
  isDisabled?: boolean
  isInvalid?: boolean
  /** Shown under the field and announced with it. Needs `isInvalid` to appear. */
  errorMessage?: string
  shouldReserveMessage?: boolean
  /** Layout and width at the call site. Not for restyling the trigger. */
  className?: string
}) {
  // A value no option carries is a URL naming something the list does not hold,
  // or a stored setting whose option has since been withdrawn. react-aria
  // answers an unmatched key with its own "Select an item", which would put
  // library boilerplate where the current value belongs, so the value is
  // carried as its own option instead. Same fallback FilterSelect makes.
  //
  // The empty string is excluded, and that is the whole reason this is not one
  // condition. Here "" means nothing is selected, not a value to preserve, so
  // synthesizing an option for it put a blank row at the top of the list that an
  // operator could pick. `FilterSelect` needs no such guard because "" is a real
  // choice there ("All", "Any price") and its callers always carry that option.
  const items: readonly SelectOption[] =
    value === "" || options.some((option) => option.value === value)
      ? options
      : [{ value, label: value }, ...options]

  const hasDescriptions = items.some((option) => option.description)

  return (
    <HeroSelect.Root
      selectedKey={value === "" ? null : optionKey(value)}
      isRequired={isRequired}
      isDisabled={isDisabled}
      isInvalid={isInvalid}
      onSelectionChange={(key) => {
        // A null key is react-aria clearing the selection. Reported as the
        // empty string rather than swallowed, because a form field can legally
        // be cleared where a filter cannot, and String(null) would put "null"
        // into the value.
        onChange(key == null ? "" : optionValue(String(key)))
      }}
      className={`flex flex-col gap-1 ${className}`}
    >
      {/* No manual "*": HeroUI marks a required field's label through CSS, so
          adding one renders two. */}
      {labelAction ? (
        <span className="flex items-center gap-1">
          <Label className="text-body">{label}</Label>
          {labelAction}
        </span>
      ) : (
        <Label className="text-body">{label}</Label>
      )}
      <HeroSelect.Trigger autoFocus={autoFocus}>
        {/* `isPlaceholder` and not a nullish check on `selectedText`, which
            reads as the same test and is not: react-aria reports an empty
            selection as `""`, so `selectedText ?? placeholder` renders a blank
            slot. HeroUI's `data-placeholder` comes off this same flag, so the
            text and the ink that marks it as a hint always agree. */}
        <HeroSelect.Value>
          {({ isPlaceholder, selectedText }) =>
            isPlaceholder ? placeholder : selectedText
          }
        </HeroSelect.Value>
        <HeroSelect.Indicator />
      </HeroSelect.Trigger>
      {/* Wider than a narrow trigger where the choices carry a sentence each. */}
      <HeroSelect.Popover className={hasDescriptions ? "min-w-72" : undefined}>
        <ListBox items={items} className="max-h-72 overflow-auto">
          {(option: SelectOption) => (
            <ListBoxItem
              id={optionKey(option.value)}
              textValue={option.label}
              aria-label={option.description ? option.label : undefined}
              isDisabled={option.isDisabled}
            >
              {option.description ? (
                // The library's description slot, so the sentence is read as
                // the option's description while `aria-label` keeps its name
                // the label alone.
                <span className="flex min-w-0 flex-col gap-0.5">
                  <span>{option.label}</span>
                  <Description className="whitespace-normal text-muted">
                    {option.description}
                  </Description>
                </span>
              ) : (
                option.label
              )}
            </ListBoxItem>
          )}
        </ListBox>
      </HeroSelect.Popover>
      <FieldMessages shouldReserve={shouldReserveMessage}>
        {description ? (
          <Description className="text-muted">{description}</Description>
        ) : null}
        {/* Not HeroUI's `FieldError`: it renders through a field's error slot,
            which `Select.Root` does not provide, so an error routed through it
            would not render at all. A span carrying the danger ink is what is
            left, and the control is marked invalid through `isInvalid` above,
            which is what assistive tech reads. */}
        {isInvalid && errorMessage ? (
          <span className="text-danger">{errorMessage}</span>
        ) : null}
      </FieldMessages>
    </HeroSelect.Root>
  )
}
