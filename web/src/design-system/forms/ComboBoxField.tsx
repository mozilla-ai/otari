import {
  ComboBox,
  Description,
  FieldError,
  Input,
  Label,
  ListBox,
  ListBoxItem,
} from "@heroui/react"
import { type ReactNode, useState } from "react"

import { ComboBoxEmpty } from "@/design-system/forms/ComboBoxEmpty"
import { FieldMessages } from "@/design-system/forms/FieldMessages"

/** One choice in a `ComboBoxField`. `isDisabled` shows a choice that exists but cannot be taken. */
export interface ComboBoxOption {
  /** The option's id: what the field reports and the key react-aria carries. Never empty: react-aria reads an empty key as "nothing selected". */
  value: string
  /** What the row and the input display. Never what the field reports. */
  label: string
  /** A second, muted line inside the row, for text that identifies the label rather than repeating it. */
  hint?: string
  isDisabled?: boolean
}

/**
 * A row's accessible name: its label, with its hint in parentheses.
 *
 * Exported for a picker that renders its own `ListBox` rows rather than going
 * through this field, so two controls an operator reads as a pair cannot
 * describe one row differently.
 */
export const comboBoxOptionText = (
  option: Pick<ComboBoxOption, "label" | "hint">,
) => (option.hint ? `${option.label} (${option.hint})` : option.label)

// `options` is the whole popover, so react-aria must not filter it again: its
// own filter reads a row's `textValue`, which is the label, and would drop a
// row the caller matched on its hint. Hoisted so the collection it feeds is not
// rebuilt on every render.
const KEEP_EVERY_OPTION = () => true

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
 * `value` is an option's `value`, as in `forms/Select`, and never a label. An
 * id and a name point at the same thing, so identity travels as the id,
 * through react-aria's `selectedKey`, and the label is only displayed: this
 * field owns the input's text and shows the matched option's label. Picking a
 * row reports its id, once. With `allowsCustomValue` the text an operator
 * types is reported as the value too, because a list that is a shortcut rather
 * than a whitelist has to let something it never offered stand; that text is
 * then itself rather than a lookup, since a label maps back to no single id
 * (two rows may share one, and one row's label may be another row's id).
 *
 * No filtering of its own: `options` is what the popover holds, and
 * `onQueryChange` is what the caller matches on. The caller matches and caps,
 * because what counts as a match differs per field (an id as well as a name, a
 * ceiling with a "showing N of M" line under it).
 */
export function ComboBoxField({
  label,
  value,
  onChange,
  onQueryChange,
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
  /** The selected option's `value`, or, where custom values are allowed, text no option carries. */
  value: string
  /** Takes the value, never an event, which is the convention every control here follows. */
  onChange: (value: string) => void
  /**
   * The input's text, for a caller that filters `options` by it. Empty once a
   * row is picked, because the field is then showing a choice rather than a
   * search. Only this field can report it: it owns the input's text.
   */
  onQueryChange?: (query: string) => void
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
  const selected = options.find((option) => option.value === value)
  // What the value reads as: the matched row's label, or the value itself,
  // which is then text no row carries.
  const shown = selected?.label ?? value

  // What is being typed, which stands in for the value's own text until a row
  // is picked or the field is committed. Undefined the rest of the time, so a
  // label the caller resolves after mount reaches the input rather than leaving
  // an id in the box.
  const [typed, setTyped] = useState<string>()

  // The value as this field last saw it, which is what tells a value the caller
  // moved from one this field reported. The first kind leaves whatever is in the
  // box stale: a list of these fields that drops a row moves a value under a
  // field that is still mounted.
  const [seen, setSeen] = useState(value)
  if (value !== seen) {
    setSeen(value)
    if (value !== typed) setTyped(undefined)
  }

  const setQuery = (next: string | undefined) => {
    setTyped(next)
    onQueryChange?.(next ?? "")
  }

  return (
    <ComboBox.Root
      allowsCustomValue={allowsCustomValue}
      // Without this, `options` going empty closes the popover, which reads as
      // "the field broke" rather than "no matches". It is also what makes the
      // empty message below reachable at all.
      allowsEmptyCollection
      menuTrigger={menuTrigger}
      defaultFilter={KEEP_EVERY_OPTION}
      // Both halves controlled, which is what keeps a pick reported once:
      // react-aria writes the picked row's text back into the input only while
      // one of the two is uncontrolled, and leaves the text here otherwise.
      inputValue={typed ?? shown}
      // The row an open list marks as selected. Nothing is selected while text
      // is being typed, because text that happens to match a row's id is still
      // text, and treating it as a pick would close the list mid-search.
      selectedKey={selected && typed === undefined ? selected.value : null}
      onInputChange={(next) => {
        setQuery(next)
        // A field offering the list as suggestions reports what was typed.
        // Elsewhere the value stays whichever row is selected, except that an
        // emptied box is how that selection is cleared.
        if (allowsCustomValue || next === "") onChange(next)
      }}
      onSelectionChange={(key) => {
        // The input goes back to showing the value, which after a pick is that
        // row's label. `key` is null when the field was committed on text no
        // row matches, which needs no report: with custom values that text is
        // already the value, and without them the value never left the row.
        setQuery(undefined)
        if (key != null) onChange(String(key))
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
              // The label alone, because this is also the text react-aria reads
              // the selection as, and it has to agree with what the input shows
              // or re-picking the selected row rewrites the input.
              textValue={option.label}
              // Spelled out, because the row has two text nodes and the name
              // computed from them runs the hint onto the end of the label with
              // no separator. The hint belongs in the name rather than being
              // dropped from it: it is what tells two rows with one label apart.
              aria-label={option.hint ? comboBoxOptionText(option) : undefined}
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
