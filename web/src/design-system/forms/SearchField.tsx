import { SearchField as HeroSearchField } from "@heroui/react"
import { useEffect, useState } from "react"

/**
 * How long after the last keystroke the field waits for the caller to hand
 * back what it reported before showing `value` instead.
 */
export const SEARCH_SETTLE_MS = 1000

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
 *
 * What was typed shows at once, without waiting for `value` to come back, so a
 * caller may commit it later, as the router does with a search param. `value`
 * still wins: a change from elsewhere replaces the text, and a keystroke the
 * caller refuses or drops is undone once typing stops.
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
  // Bound to `value` straight, the field would revert for the render between a
  // keystroke and the caller's commit, and that revert drops the caret to the
  // end. `reported` holds what was handed to the caller and not yet seen back:
  // a `value` among them is the caller catching up, even to an older keystroke,
  // and any other `value` is a change from elsewhere.
  const [draft, setDraft] = useState(value)
  const [shown, setShown] = useState(value)
  const [reported, setReported] = useState<readonly string[]>([])
  if (value !== shown) {
    setShown(value)
    const caughtUp = reported.indexOf(value)
    if (caughtUp === -1) {
      setDraft(value)
      setReported([])
    } else {
      setReported(reported.slice(caughtUp + 1))
    }
  }
  // A report never seen back was refused, or dropped for a newer navigation
  // whose `value` happens to be the one already shown.
  useEffect(() => {
    if (reported.length === 0) return
    const timer = setTimeout(() => {
      setDraft(value)
      setReported([])
    }, SEARCH_SETTLE_MS)
    return () => clearTimeout(timer)
  }, [reported, value])
  return (
    <HeroSearchField.Root
      aria-label={label}
      value={draft}
      onChange={(next) => {
        setDraft(next)
        setReported((current) => [...current, next])
        onChange(next)
      }}
      isDisabled={isDisabled}
      maxLength={maxLength}
      className={className}
    >
      <HeroSearchField.Group>
        <HeroSearchField.SearchIcon />
        <HeroSearchField.Input placeholder={placeholder} />
        <HeroSearchField.ClearButton aria-label="Clear search" />
      </HeroSearchField.Group>
    </HeroSearchField.Root>
  )
}
