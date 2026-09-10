import { type ReactNode, useId, useRef, useState } from "react"
import { FiCheck, FiChevronDown } from "react-icons/fi"

import { DismissChip } from "../indicators/DismissChip"
import { FieldMessages } from "./FieldMessages"
import { INPUT_CLASS } from "./inputClass"

export interface MultiSelectOption {
  id: string
  label: string
}

/** Six rows at 36px, which is where the list starts scrolling. */
const MAX_VISIBLE_ROWS = 6

/**
 * Pick several from a searchable list, without the field moving.
 *
 * The form-grade half of the pair `forms.md` describes: `FilterMultiComboBox` is
 * the toolbar one and stays. Two things make this the form's, and both are the
 * bug it was built for. The search field is at the top and never moves, because
 * the chips render *below* it, so a growing selection pushes down rather than
 * shoving the field out from under the pointer. And a selected option stays in
 * the list, checked, so the list answers "who is in" rather than only "who is
 * left"; pressing it again removes it, and the order never re-sorts on a pick.
 *
 * A combo box with a listbox popup, hand-rolled to the APG pattern rather than
 * built on HeroUI's `ComboBox`: that one selects a single value and closes, and
 * every behavior here that matters (toggle, stay listed, Escape closing only the
 * popover) is a departure from it. Focus stays in the input the whole time and
 * the active option travels by `aria-activedescendant`, which is what lets the
 * arrows move a highlight while the query keeps taking keystrokes.
 */
export function MultiSelect({
  label,
  description,
  options,
  value,
  onChange,
  isInvalid,
  errorMessage,
  reserveMessage,
  searchPlaceholder = "Search…",
  emptyMessage = "Nothing to choose from.",
  noMatchesMessage = "Nothing matches what you typed.",
  countNoun = "selected",
}: {
  label: string
  description?: ReactNode
  options: readonly MultiSelectOption[]
  value: readonly string[]
  onChange: (next: string[]) => void
  isInvalid?: boolean
  errorMessage?: ReactNode
  reserveMessage?: boolean
  /** The closed field's text while nothing is picked. */
  searchPlaceholder?: string
  /** Shown in the popover when there is nothing to offer at all. */
  emptyMessage?: ReactNode
  /** Shown when the query matches none of the options. */
  noMatchesMessage?: ReactNode
  /** Pluralized into the closed field and the footer: "6 people assigned". */
  countNoun?: string
}) {
  const [query, setQuery] = useState("")
  const [isOpen, setIsOpen] = useState(false)
  const [activeIndex, setActiveIndex] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const listId = useId()
  const inputId = useId()
  const optionId = (index: number) => `${listId}-option-${index}`

  const needle = query.trim().toLowerCase()
  // Filtered, never re-ordered: a selected row keeps its place, so a second
  // press lands on the row the first one did.
  const matches = options.filter(
    (option) =>
      needle === "" ||
      option.id.toLowerCase().includes(needle) ||
      option.label.toLowerCase().includes(needle),
  )
  const selectedMatches = matches.filter((option) =>
    value.includes(option.id),
  ).length
  const labelOf = (id: string) =>
    options.find((option) => option.id === id)?.label ?? id

  const toggle = (id: string) => {
    onChange(
      value.includes(id)
        ? value.filter((current) => current !== id)
        : [...value, id],
    )
    // The query survives a pick. Clearing it would refill the list under the
    // pointer, which is the same movement the chips were moved to avoid.
    inputRef.current?.focus()
  }

  const open = () => {
    setIsOpen(true)
    setActiveIndex(0)
  }

  const onKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault()
      if (!isOpen) {
        open()
        return
      }
      if (matches.length === 0) return
      const step = event.key === "ArrowDown" ? 1 : -1
      setActiveIndex(
        (current) => (current + step + matches.length) % matches.length,
      )
      return
    }
    if (event.key === "Enter" || event.key === " ") {
      // Space only toggles when the query is empty; otherwise it is a character
      // somebody is typing into a name.
      if (event.key === " " && query !== "") return
      if (!isOpen) return
      const option = matches[activeIndex]
      if (!option) return
      event.preventDefault()
      toggle(option.id)
      return
    }
    if (event.key === "Escape" && isOpen) {
      // Stopped here rather than allowed to bubble: this dialog's own Escape
      // closes the dialog, and dismissing a popover must not also throw away
      // the form behind it.
      event.preventDefault()
      event.stopPropagation()
      setIsOpen(false)
      return
    }
    if (event.key === "Backspace" && query === "" && value.length > 0) {
      onChange(value.slice(0, -1))
    }
  }

  return (
    <div className="flex flex-col gap-1.5">
      {/* A real label with `htmlFor`, not HeroUI's `Label`: that one resolves
          its target through a field context this control is not inside, so it
          would render the words and name nothing. The visible text IS the
          accessible name here, which is why there is no `aria-label` beside
          it. */}
      <label htmlFor={inputId} className="text-body">
        {label}
      </label>
      {description ? <p className="text-caption">{description}</p> : null}
      {/* `relative` so the popover hangs off the field rather than off the
          dialog, and the chips below it stay in flow. */}
      <div className="relative flex flex-col gap-1.5">
        <div className="relative">
          <input
            ref={inputRef}
            id={inputId}
            role="combobox"
            aria-expanded={isOpen}
            aria-controls={listId}
            aria-autocomplete="list"
            aria-activedescendant={
              isOpen && matches[activeIndex] ? optionId(activeIndex) : undefined
            }
            aria-invalid={isInvalid}
            className={`${INPUT_CLASS} w-full pr-9`}
            // The closed field says how many are in rather than staying blank,
            // because the chips below it can be scrolled past in a long form.
            placeholder={
              value.length === 0
                ? searchPlaceholder
                : `${value.length} ${countNoun}`
            }
            value={query}
            autoComplete="off"
            onChange={(event) => {
              setQuery(event.target.value)
              setActiveIndex(0)
              setIsOpen(true)
            }}
            onFocus={open}
            onBlur={(event) => {
              // Only when focus actually left the control: a press on an option
              // blurs the input and must not close the list before the press
              // lands.
              if (
                !event.currentTarget.parentElement?.parentElement?.contains(
                  event.relatedTarget,
                )
              ) {
                setIsOpen(false)
              }
            }}
            onKeyDown={onKeyDown}
          />
          <FiChevronDown
            aria-hidden
            className="text-muted pointer-events-none absolute top-1/2 right-3 size-3 -translate-y-1/2"
          />
        </div>
        {isOpen ? (
          <div className="border-control-border bg-surface absolute top-full right-0 left-0 z-10 mt-1 border">
            {/* Divs rather than a ul/li pair: the roles are what carry the
                semantics here, and a list element with an interactive role is
                both a lint error and a second, conflicting announcement. The
                chip row below IS a real list, because that one is a list. */}
            <div
              id={listId}
              role="listbox"
              aria-multiselectable
              // Named apart from the field and from the chip row: three things
              // carrying one name is three things a screen reader cannot tell
              // apart, and it is what a query for the field then finds.
              aria-label={`${label}, options`}
              className="overflow-y-auto"
              style={{ maxHeight: `${MAX_VISIBLE_ROWS * 36}px` }}
            >
              {matches.length === 0 ? (
                <p className="text-caption px-2.5 py-2">
                  {options.length === 0 ? emptyMessage : noMatchesMessage}
                </p>
              ) : (
                matches.map((option, index) => {
                  const isSelected = value.includes(option.id)
                  return (
                    <div
                      key={option.id}
                      id={optionId(index)}
                      role="option"
                      aria-selected={isSelected}
                      // Not tabbable, and deliberately not focused either: the
                      // input keeps focus and `aria-activedescendant` points at
                      // the active row. -1 is what the option needs to be a
                      // legal target for that.
                      tabIndex={-1}
                      // Pressed rather than clicked through a button: the row is
                      // the option, and a button inside it would be a second
                      // stop for a keyboard that is already driving the list
                      // from the input.
                      onMouseDown={(event) => {
                        // Before blur, so the press is not lost to the list
                        // closing under it.
                        event.preventDefault()
                        toggle(option.id)
                      }}
                      onMouseEnter={() => setActiveIndex(index)}
                      className={`flex h-9 cursor-pointer items-center gap-2.5 border-border-subtle px-2.5 text-sm not-first:border-t ${
                        // `surface-alt`, not `surface-muted`: the two resolve
                        // to the same value, and only this one is registered in
                        // @theme, so only this one emits a rule.
                        isSelected ? "bg-surface-alt" : ""
                      } ${index === activeIndex ? "bg-surface-subtle" : ""}`}
                    >
                      <span className="flex size-4 shrink-0 items-center justify-center">
                        {isSelected ? (
                          <FiCheck aria-hidden className="text-accent size-3" />
                        ) : null}
                      </span>
                      {option.label}
                    </div>
                  )
                })
              )}
            </div>
            <div className="border-border text-mono-caption text-subtle flex h-8 items-center justify-between border-t px-2.5">
              <span aria-live="polite">
                {selectedMatches} of {matches.length} matches selected ·{" "}
                {value.length} {countNoun}
              </span>
              <span>ESC closes</span>
            </div>
          </div>
        ) : null}
        {value.length > 0 ? (
          // Below the field, which is the whole point: a growing selection
          // pushes down rather than moving the control it grew from.
          <ul
            aria-label={`${label}, selected`}
            className="flex list-none flex-wrap gap-x-2 gap-y-1.5 pt-0.5"
          >
            {value.map((id) => (
              <li key={id}>
                <DismissChip
                  value={labelOf(id)}
                  onDismiss={() => toggle(id)}
                  dismissLabel={`Remove ${labelOf(id)}`}
                />
              </li>
            ))}
          </ul>
        ) : null}
      </div>
      <FieldMessages reserve={reserveMessage}>
        {isInvalid && errorMessage ? (
          <span className="text-danger">{errorMessage}</span>
        ) : null}
      </FieldMessages>
    </div>
  )
}
