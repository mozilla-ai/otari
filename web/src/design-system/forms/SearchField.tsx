import { SearchField as HeroSearchField } from "@heroui/react"
import { useState } from "react"

/**
 * A search box: a magnifier, a field, and a clear button once there is
 * something to clear.
 *
 * `type="search"` rather than a text input with an icon beside it, which is
 * what earns the two behaviors a hand-rolled version does not get: Escape
 * clears the field, and the browser offers previous searches for this field
 * rather than the page's generic autofill.
 *
 * Filtering is the caller's, always. This reports what was typed and nothing
 * else, so a page can debounce it, push it into the URL, or hand it to a query.
 * A search box that filtered its own siblings would need to know what they are.
 */
export function SearchField({
  value,
  onChange,
  label,
  placeholder = "Search",
  isDisabled,
  maxLength,
  className = "",
}: {
  value: string
  onChange: (value: string) => void
  /**
   * The accessible name. Required, because the placeholder is not one: it
   * disappears the moment somebody types, taking the field's only name with it.
   */
  label: string
  placeholder?: string
  isDisabled?: boolean
  maxLength?: number
  className?: string
}) {
  // The field answers each keystroke from its own draft. A caller that commits
  // in a transition, as the router does with a search param, hands the old
  // `value` back for one render after the keystroke, and a field bound to it
  // straight reverts and drops the caret to the end. A `value` changed from
  // elsewhere, such as a link that drops the param, still replaces the draft.
  const [draft, setDraft] = useState(value)
  const [shown, setShown] = useState(value)
  if (value !== shown) {
    setShown(value)
    setDraft(value)
  }
  return (
    <HeroSearchField.Root
      aria-label={label}
      value={draft}
      onChange={(next) => {
        setDraft(next)
        onChange(next)
      }}
      isDisabled={isDisabled}
      maxLength={maxLength}
      className={className}
    >
      <HeroSearchField.Group>
        <HeroSearchField.SearchIcon />
        <HeroSearchField.Input placeholder={placeholder} />
        <HeroSearchField.ClearButton />
      </HeroSearchField.Group>
    </HeroSearchField.Root>
  )
}
