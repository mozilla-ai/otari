import { type ReactNode, useEffect, useId, useRef, useState } from "react"
import { FiCheck, FiChevronDown } from "react-icons/fi"

import { DismissChip } from "../indicators/DismissChip"
import { FieldMessages } from "./FieldMessages"
import { INPUT_CLASS } from "./inputClass"

export interface MultiSelectOption {
  id: string
  label: string
  /**
   * A second, muted line inside the row, for text that identifies the label
   * rather than repeating it, such as the id a person is billed under. It joins
   * the row's accessible name, since it is what tells two rows with one label
   * apart. Same shape and same treatment as `ComboBoxField`'s.
   */
  hint?: string
}

/** How many matches are rendered at once, however many the query reaches. */
const DEFAULT_MAX_VISIBLE = 50

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
  countNoun = { one: "selected", other: "selected" },
  maxVisible = DEFAULT_MAX_VISIBLE,
  autoFocus,
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
  /**
   * Pluralized into the closed field and the footer: "1 person assigned",
   * "6 people assigned". A pair rather than one string, because interpolating
   * one noun into both counts is how "1 people assigned" ships.
   */
  countNoun?: { one: string; other: string }
  /**
   * How many matches to render. The filter runs over every option; this caps
   * what is mounted, because a deployment's roster is unbounded and the popover
   * is inside a modal. The footer says when it is capping.
   */
  maxVisible?: number
  /** A form's first field takes this; see feedback.md. */
  autoFocus?: boolean
}) {
  const [query, setQuery] = useState("")
  const [isOpen, setIsOpen] = useState(false)
  const [activeIndex, setActiveIndex] = useState(0)
  // The control's own box, which is what "focus left this control" means. Held
  // as a ref rather than walked up from the input, so wrapping the input in one
  // more div cannot silently close the popover before an option press lands.
  const wrapperRef = useRef<HTMLDivElement>(null)
  // The list, so the active option can be scrolled into view: the popover shows
  // six rows and the cap renders fifty, so the highlight walks off the bottom.
  const listRef = useRef<HTMLDivElement>(null)
  const listId = useId()
  const inputId = useId()
  const descriptionId = useId()
  const errorId = useId()
  const optionId = (index: number) => `${listId}-option-${index}`

  // The popover shows six rows and renders up to `maxVisible`, so the active
  // row walks past the fold on its own. Scrolled here rather than by the row,
  // because only the list knows which one is active.
  useEffect(() => {
    if (!isOpen) return
    // The id is spelled here rather than through `optionId`, which is a fresh
    // arrow every render and so cannot be a dependency.
    const active = listRef.current?.querySelector(
      `[id="${listId}-option-${activeIndex}"]`,
    )
    // Guarded because jsdom implements no layout and so no `scrollIntoView`;
    // the tests below drive the arrows and would throw on the first press.
    if (
      active instanceof HTMLElement &&
      typeof active.scrollIntoView === "function"
    ) {
      active.scrollIntoView({ block: "nearest" })
    }
  }, [isOpen, activeIndex, listId])

  const needle = query.trim().toLowerCase()
  // Filtered, never re-ordered: a selected row keeps its place, so a second
  // press lands on the row the first one did.
  const reached = options.filter(
    (option) =>
      needle === "" ||
      option.id.toLowerCase().includes(needle) ||
      option.label.toLowerCase().includes(needle) ||
      (option.hint?.toLowerCase().includes(needle) ?? false),
  )
  // Capped after the filter, never before: the query has to see every option,
  // and only the rendering is bounded.
  const matches = reached.slice(0, maxVisible)
  const capped = reached.length > matches.length
  const selectedMatches = matches.filter((option) =>
    value.includes(option.id),
  ).length
  const labelOf = (id: string) =>
    options.find((option) => option.id === id)?.label ?? id
  const counted = (n: number) =>
    `${n} ${n === 1 ? countNoun.one : countNoun.other}`

  // The query survives a pick. Clearing it would refill the list under the
  // pointer, which is the same movement the chips were moved to avoid.
  //
  // No refocus here: a chip's Remove button calls this while the popover is
  // closed, and focusing the input would reopen the list over the chip row. An
  // option press keeps focus anyway, because its `onMouseDown` prevents the
  // default that would have moved it.
  const toggle = (id: string) => {
    onChange(
      value.includes(id)
        ? value.filter((current) => current !== id)
        : [...value, id],
    )
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
      // Before the row lookup, not after: an open combo box owns Enter whether
      // or not it has a row to give, and this input sits inside a real form, so
      // falling through submits it. Stopped as well as prevented, because
      // `FormDialog` submits on Cmd/Ctrl+Enter from anywhere inside the form:
      // without this, one gesture toggles a row and posts the selection from
      // before that toggle.
      event.preventDefault()
      event.stopPropagation()
      const option = matches[activeIndex]
      if (!option) return
      toggle(option.id)
      return
    }
    if (event.key === "Tab" && isOpen) {
      // Closed rather than trapped: the APG pattern dismisses the popup on Tab,
      // and leaving it open puts the focus ring on a chip's Remove button drawn
      // underneath the panel.
      setIsOpen(false)
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
      {/* `relative` so the popover hangs off the field rather than off the
          dialog, and the chips below it stay in flow. */}
      <div ref={wrapperRef} className="relative flex flex-col gap-1.5">
        <div className="relative">
          <input
            id={inputId}
            role="combobox"
            aria-expanded={isOpen}
            aria-controls={listId}
            aria-autocomplete="list"
            aria-activedescendant={
              isOpen && matches[activeIndex] ? optionId(activeIndex) : undefined
            }
            aria-invalid={isInvalid}
            // Wired by hand, because this does not go through HeroUI's
            // Description and FieldError slots: without it a screen reader
            // says "invalid" and never says why.
            // Whichever line is actually rendered, since the error replaces
            // the description rather than joining it.
            aria-describedby={
              isInvalid && errorMessage
                ? errorId
                : description
                  ? descriptionId
                  : undefined
            }
            className={`${INPUT_CLASS} w-full pr-9`}
            // The closed field says how many are in rather than staying blank,
            // because the chips below it can be scrolled past in a long form.
            placeholder={
              value.length === 0 ? searchPlaceholder : counted(value.length)
            }
            value={query}
            autoComplete="off"
            // biome-ignore lint/a11y/noAutofocus: a form's first field takes it; see feedback.md
            autoFocus={autoFocus}
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
              if (!wrapperRef.current?.contains(event.relatedTarget)) {
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
              ref={listRef}
              id={listId}
              role="listbox"
              aria-multiselectable
              // Named apart from the field and from the chip row: three things
              // carrying one name is three things a screen reader cannot tell
              // apart, and it is what a query for the field then finds.
              aria-label={`${label}, options`}
              // Six rows before it scrolls. Each is 44px, the touch floor
              // (motion-and-access.md: "44px is the floor, everywhere"), and
              // all but the first carry a 1px rule, so six of them are
              // 6 x 44 + 5 = 269px. `max-h-[16.8125rem]` rather than the 264px
              // the rows alone would suggest, which clipped the sixth by 5px.
              // A class rather than an inline style: it is a constant.
              className="max-h-[16.8125rem] overflow-y-auto"
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
                      aria-label={
                        option.hint
                          ? `${option.label} (${option.hint})`
                          : undefined
                      }
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
                      // One background, picked here. Two classes on one element
                      // are resolved by Tailwind's emitted order rather than by
                      // the order they are written in, which `inputClass.ts`
                      // documents as a hazard. `surface-alt` rather than
                      // `surface-muted`: the two resolve to the same value and
                      // only this one is registered in @theme.
                      className={`flex min-h-11 cursor-pointer items-center gap-2.5 border-border-subtle px-2.5 text-sm not-first:border-t ${
                        index === activeIndex
                          ? "bg-surface-subtle"
                          : isSelected
                            ? "bg-surface-alt"
                            : ""
                      }`}
                    >
                      <span className="flex size-4 shrink-0 items-center justify-center">
                        {isSelected ? (
                          <FiCheck aria-hidden className="text-accent size-3" />
                        ) : null}
                      </span>
                      {/* Spelled out when there is a hint, because the row has
                          two text nodes and the name computed from them runs
                          the hint onto the end of the label with no separator.
                          The hint belongs in the name rather than being dropped
                          from it: it is what tells two rows with one label
                          apart. `ComboBoxField` does this the same way. */}
                      <span className="flex min-w-0 flex-col">
                        <span className="truncate">{option.label}</span>
                        {option.hint ? (
                          <span className="text-caption text-subtle truncate">
                            {option.hint}
                          </span>
                        ) : null}
                      </span>
                    </div>
                  )
                })
              )}
            </div>
            <div className="border-border text-mono-caption text-subtle flex h-8 items-center justify-between border-t px-2.5">
              {/* Two facts, two clauses. Nested, they read as one garbled
                  sentence: "0 of 3 matches selected of 8". */}
              <span>
                {capped
                  ? `Showing ${matches.length} of ${reached.length} · `
                  : ""}
                {selectedMatches} of {matches.length} selected here ·{" "}
                {counted(value.length)}
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
      {/* Mounted whatever the popover is doing. A live region inserted with
          its content already in it announces nothing, so a region that only
          exists while the list is open is silent on the first count and absent
          when a chip is removed from the closed state. */}
      <span aria-live="polite" className="sr-only">
        {counted(value.length)}
      </span>
      {/* One rung, below the control, the way `Field` does it: the error
          replaces the description rather than adding a row. Below the chips
          rather than above the field, because that is where a `Field` puts it
          and a dialog full of `Field`s should not have one control speaking
          from somewhere else. */}
      <FieldMessages reserve={reserveMessage}>
        {isInvalid && errorMessage ? (
          <span id={errorId} className="text-danger">
            {errorMessage}
          </span>
        ) : description ? (
          <span id={descriptionId} className="text-muted">
            {description}
          </span>
        ) : null}
      </FieldMessages>
    </div>
  )
}
