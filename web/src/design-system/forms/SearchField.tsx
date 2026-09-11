import { SearchField as HeroSearchField } from "@heroui/react"

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
  className?: string
}) {
  return (
    <HeroSearchField.Root
      aria-label={label}
      value={value}
      onChange={onChange}
      isDisabled={isDisabled}
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
