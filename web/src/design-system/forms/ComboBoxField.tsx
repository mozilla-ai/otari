import {
  ComboBox,
  Description,
  FieldError,
  Input,
  Label,
  ListBox,
  ListBoxItem,
} from "@heroui/react"
import { type ReactNode, useRef } from "react"

import { ComboBoxEmpty } from "@/design-system/forms/ComboBoxEmpty"
import { FieldMessages } from "@/design-system/forms/FieldMessages"

/** One choice in a `ComboBoxField`. `isDisabled` shows a choice that exists but cannot be taken. */
export interface ComboBoxOption {
  /** What the field reports, and the key react-aria carries. Never empty: react-aria reads an empty key as "nothing selected". */
  value: string
  label: string
  /** A second, muted line inside the row, for text that identifies the label rather than repeating it. */
  hint?: string
  isDisabled?: boolean
}

const optionText = (option: ComboBoxOption) =>
  option.hint ? `${option.label} (${option.hint})` : option.label

/**
 * One of a set, searchable, in a form the operator submits.
 *
 * The combo box counterpart to `forms/Select`, and the prop vocabulary is
 * deliberately Select's wherever the two mean the same thing: an operator of
 * this codebase reads the "Signatures" table in `web/design/forms.md` as one
 * vocabulary, and a control that spelled `onChange` or `errorMessage`
 * differently would cost them that. A combo box is not a longer select, so
 * three things are its own: free text may be allowed, the list opens on typing
 * or on focus, and the list can legitimately be empty.
 *
 * That last one is why this exists rather than each page building its own. A
 * combo box keeps its menu open on an empty collection, so that a query
 * matching nothing does not read as a field that broke; the cost is a popover
 * with nothing in it, which reads as a field that broke for a different reason.
 * `ComboBoxEmpty` is the sentence that tells those apart, and there is no way
 * to render this control without one.
 *
 * `value` is the input's text rather than a chosen key, which is what lets a
 * caller with `allowsCustomValue` submit something the list never offered.
 * Picking a row reports that option's `value`, and reports it once: react-aria
 * echoes the row's display text into the input afterwards, and that echo is
 * swallowed here rather than forwarded. So `onChange` carries a key when a row
 * was picked and text when text was typed, and never a label. See the handlers
 * for why the caller cannot be the one to sort those out.
 *
 * No filtering of its own: `options` is what the popover holds. The caller
 * matches and caps, because what counts as a match differs per field (an id as
 * well as a name, a ceiling with a "showing N of M" line under it).
 */
export function ComboBoxField({
  label,
  value,
  onChange,
  options,
  description,
  placeholder,
  isRequired,
  isDisabled,
  isInvalid,
  errorMessage,
  reserveMessage,
  className = "",
  allowsCustomValue,
  autoFocus,
  menuTrigger = "focus",
  shouldSelectOnFocus,
  isSourceEmpty,
  emptyMessage,
  noMatchesMessage,
}: {
  label: ReactNode
  /** The input's text. Not a key: a field allowing custom values holds text no option carries. */
  value: string
  /** Takes the value, never an event, which is the convention every control here follows. */
  onChange: (value: string) => void
  /** Already filtered and capped by the caller, whose match rules and ceiling are its own. */
  options: readonly ComboBoxOption[]
  description?: ReactNode
  /** Shown while the field is empty. An example, never the label. */
  placeholder?: string
  isRequired?: boolean
  isDisabled?: boolean
  isInvalid?: boolean
  /** Shown under the field and announced with it. Needs `isInvalid` to appear. */
  errorMessage?: string
  reserveMessage?: boolean
  /** Layout and width at the call site. Not for restyling the field. */
  className?: string
  /** Offer the list as suggestions rather than as a whitelist, so anything typed stands. */
  allowsCustomValue?: boolean
  autoFocus?: boolean
  /**
   * When the list opens. `"focus"` suits a pick-from-a-list field; `"input"` is
   * for one that is autofocused, since react-aria marks everything outside an
   * open popover aria-hidden and a list open on arrival hides the rest of the form.
   */
  menuTrigger?: "focus" | "input"
  /** Select the text on focus, so typing replaces the shown selection instead of appending to it. */
  shouldSelectOnFocus?: boolean
  /** True when `options` is empty whatever is typed, which is what picks between the two empty sentences. */
  isSourceEmpty?: boolean
  /** What the empty popover says while `isSourceEmpty`. Say what would fill the list. */
  emptyMessage?: ReactNode
  /** What it says when the source has options and the query matched none. */
  noMatchesMessage?: ReactNode
}) {
  // Not state: nothing renders from it, and a re-render between the pick and
  // its echo would drop it.
  const echoRef = useRef<string | undefined>(undefined)

  return (
    <ComboBox.Root
      allowsCustomValue={allowsCustomValue}
      // Without this, `options` going empty closes the popover, which reads as
      // "the field broke" rather than "no matches". It is also what makes the
      // empty message below reachable at all.
      allowsEmptyCollection
      menuTrigger={menuTrigger}
      inputValue={value}
      // The echo, and why it is caught here. react-aria reports a pick through
      // `onSelectionChange` and then writes that row's display text into the
      // input, firing `onInputChange` with a label where the previous call
      // carried a key. A caller that forwarded both would have to turn the
      // label back into a key, and two rows may share a label, or one row's
      // label may be another row's value, so that mapping can land on the wrong
      // row. Here the picked option is in hand, so the echo is recognized by
      // identity and dropped. Anything else the operator types passes through.
      onInputChange={(next) => {
        const echoed = echoRef.current
        echoRef.current = undefined
        if (echoed !== undefined && next === echoed) return
        onChange(next)
      }}
      onSelectionChange={(key) => {
        if (key == null) return
        const picked = options.find((option) => option.value === String(key))
        echoRef.current = picked ? optionText(picked) : undefined
        onChange(String(key))
      }}
      isRequired={isRequired}
      isDisabled={isDisabled}
      isInvalid={isInvalid}
      // Bounded rather than stretching across a wide form, so the field and its
      // trigger stay within easy reach.
      className={`flex max-w-md flex-col gap-1 ${className}`}
    >
      {/* No manual "*": HeroUI marks a required field's label through CSS, so
          adding one renders two. */}
      <Label className="text-body">{label}</Label>
      <ComboBox.InputGroup>
        <Input
          placeholder={placeholder}
          autoFocus={autoFocus}
          // A picker is never a credential field, so a password manager
          // offering to fill it is wrong at every call site rather than at some.
          autoComplete="off"
          data-1p-ignore
          data-lpignore="true"
          onFocus={
            shouldSelectOnFocus
              ? (event) => event.currentTarget.select()
              : undefined
          }
        />
        <ComboBox.Trigger />
      </ComboBox.InputGroup>
      <ComboBox.Popover>
        <ListBox
          items={options}
          className="max-h-72 overflow-auto"
          renderEmptyState={() => (
            <ComboBoxEmpty
              isSourceEmpty={isSourceEmpty}
              emptyMessage={emptyMessage}
              noMatchesMessage={noMatchesMessage}
            />
          )}
        >
          {(option: ComboBoxOption) => (
            <ListBoxItem
              id={option.value}
              textValue={optionText(option)}
              // Spelled out, because the row has two text nodes and the name
              // computed from them runs the hint onto the end of the label with
              // no separator. The hint belongs in the name rather than being
              // dropped from it: it is what tells two rows with one label apart.
              aria-label={option.hint ? optionText(option) : undefined}
              isDisabled={option.isDisabled}
            >
              <span className="flex flex-col">
                <span>{option.label}</span>
                {option.hint ? (
                  <span className="text-caption text-subtle">
                    {option.hint}
                  </span>
                ) : null}
              </span>
            </ListBoxItem>
          )}
        </ListBox>
      </ComboBox.Popover>
      {/* Reserved even when silent, so this control lines up with a `Field`
          beside it in a row. The description goes through HeroUI's own slot,
          which is what wires it to the input via aria-describedby; a bare node
          there leaves the combo box reporting no description at all. */}
      <FieldMessages reserve={reserveMessage}>
        {description ? (
          <Description className="text-muted">{description}</Description>
        ) : null}
        {errorMessage ? (
          <FieldError className="text-danger">{errorMessage}</FieldError>
        ) : null}
      </FieldMessages>
    </ComboBox.Root>
  )
}
