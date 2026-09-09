import { Label, ListBox, ListBoxItem, Select } from "@heroui/react"

// react-aria reads an empty key as "nothing selected", and "" is a real filter
// value here ("All", "Any price"), so every option key carries this prefix and
// it is stripped back off on the way out. Both directions go through these two,
// so the prefix is written down once.
const OPTION_KEY_PREFIX = "v:"
const optionKey = (value: string) => `${OPTION_KEY_PREFIX}${value}`
const optionValue = (key: string) => key.slice(OPTION_KEY_PREFIX.length)

// Filter dropdown for page filter bars. On HeroUI's Select rather than a native
// <select> because a native one draws its menu *over* the control on macOS,
// covering the button that opened it; this one is a popover anchored under the
// trigger. Pass `label` for a visible label, or `ariaLabel` alone for a compact
// control, and `id` when an outside <label htmlFor> points at the trigger.

export function FilterSelect({
  id,
  label,
  ariaLabel,
  value,
  onChange,
  options,
  disabled,
}: {
  id?: string
  label?: string
  ariaLabel?: string
  value: string
  onChange: (value: string) => void
  options: { value: string; label: string }[]
  disabled?: boolean
}) {
  // A value no option carries is a URL naming something the list does not hold
  // (`/activity?status=bogus`) or a drill-down into a key with no rows in the
  // window. react-aria answers an unmatched key with its own "Select an item",
  // which would put library boilerplate where the applied filter belongs, so
  // the value is carried as its own option instead: the filter bar says what is
  // actually filtering, the same fallback the pages' own chips make with
  // `?? value`. A call site whose default is missing from its own options would
  // land here too, which is a bug at the call site rather than a shape to
  // design around; every list today carries its own.
  const items = options.some((option) => option.value === value)
    ? options
    : [{ value, label: value }, ...options]
  return (
    <Select.Root
      aria-label={label ? undefined : ariaLabel}
      isDisabled={disabled}
      selectedKey={optionKey(value)}
      // A null key is react-aria clearing the selection, which no filter here
      // asks for: reporting it would push the strip of a non-string ("ll") into
      // the filter, and on a URL-backed page into the query string with it.
      onSelectionChange={(key) => {
        if (key != null) onChange(optionValue(String(key)))
      }}
    >
      {label ? <Label className="text-caption">{label}</Label> : null}
      <Select.Trigger id={id}>
        <Select.Value />
        <Select.Indicator />
      </Select.Trigger>
      <Select.Popover>
        <ListBox items={items} className="max-h-72 overflow-auto">
          {(option: { value: string; label: string }) => (
            <ListBoxItem id={optionKey(option.value)} textValue={option.label}>
              {option.label}
            </ListBoxItem>
          )}
        </ListBox>
      </Select.Popover>
    </Select.Root>
  )
}

// A type-to-filter combobox for page filter bars, accumulating a set of values.
// The option list is large (users, models), so a native <select> with thousands of
// <option>s is unusable and this narrows it as you type; and the question a usage
// view answers is usually a comparison ("these three models"), not a single choice.
// Picking an option adds it and clears the query, and the list stays open on the remaining
// options so a run of selections takes one gesture; picked options drop out of it.
// Removal lives with the page's filter chips (one per value) rather than a second
// chip row here, so the applied set is visible whether or not the picker is open.
// Dismiss the list (Escape, or a click outside) before reaching for the page
// behind it: it is an overlay, so it holds focus while open.
