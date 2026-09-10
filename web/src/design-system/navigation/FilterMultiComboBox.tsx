import { ComboBox, Input, Label, ListBox, ListBoxItem } from "@heroui/react"
import type { KeyboardEvent as ReactKeyboardEvent } from "react"
import { useState } from "react"

import { ComboBoxEmpty } from "@/design-system/forms/ComboBoxEmpty"

export function FilterMultiComboBox({
  label,
  values,
  onChange,
  options,
  placeholder,
  maxVisible = 50,
  maxValues = 50,
  allowsCustom = false,
}: {
  label: string
  values: string[]
  onChange: (values: string[]) => void
  options: { value: string; label: string }[]
  // Shown while nothing is picked (e.g. "All users"); once something is, the
  // input reports the size of the selection instead.
  placeholder?: string
  maxVisible?: number
  // Ceiling on the selection, matching what the analytics endpoints accept for
  // one repeatable filter. Stopping here keeps a 51st pick from failing every
  // query on the page with a 422 the operator cannot read.
  maxValues?: number
  // When true, Enter adds whatever was typed, so a filter whose value space is not
  // enumerable (any model name the log might hold, not just the ones a windowed
  // suggestion list knows) can still be filtered on. The options stay suggestions.
  allowsCustom?: boolean
}) {
  const [text, setText] = useState("")

  const atLimit = values.length >= maxValues
  const query = text.trim().toLowerCase()
  const visible = options
    .filter((o) => !values.includes(o.value))
    .filter(
      (o) =>
        !query ||
        o.value.toLowerCase().includes(query) ||
        o.label.toLowerCase().includes(query),
    )
    .slice(0, maxVisible)

  const add = (value: string) => {
    if (atLimit || values.includes(value)) return
    onChange([...values, value])
  }

  // Free text commits on Enter: react-aria fires no selection for a value that is
  // not in the list, so the key event is the only signal. Skipped while an option is
  // highlighted (aria-activedescendant), because that Enter belongs to the option
  // and committing the partial query beside it would add two values from one press.
  const onInputKeyDown = (event: ReactKeyboardEvent<HTMLInputElement>) => {
    if (!allowsCustom || event.key !== "Enter") return
    if (event.currentTarget.getAttribute("aria-activedescendant")) return
    const typed = text.trim()
    if (!typed) return
    add(typed)
    setText("")
  }

  return (
    <ComboBox.Root
      allowsEmptyCollection
      allowsCustomValue={allowsCustom}
      menuTrigger="focus"
      inputValue={text}
      onInputChange={setText}
      // Never a committed selection of its own: the picked values live in
      // `values`, so the input stays a search box.
      selectedKey={null}
      // At the ceiling the remaining options are offered but inert, so the list
      // reads as "full" rather than silently swallowing a click.
      disabledKeys={atLimit ? visible.map((o) => o.value) : []}
      onSelectionChange={(key) => {
        if (key == null) return
        add(String(key))
        setText("")
      }}
      className="flex flex-col gap-1"
    >
      <Label className="text-caption">{label}</Label>
      <ComboBox.InputGroup>
        <Input
          placeholder={
            values.length === 0
              ? placeholder
              : `${values.length} selected${atLimit ? " (max)" : ""}`
          }
          autoComplete="off"
          onKeyDown={onInputKeyDown}
        />
        <ComboBox.Trigger />
      </ComboBox.InputGroup>
      <ComboBox.Popover>
        <ListBox
          items={visible}
          className="max-h-72 overflow-auto"
          // A filter still owes the operator a reason for an empty menu, even
          // though it never speaks a validation message.
          renderEmptyState={() => (
            <ComboBoxEmpty
              isSourceEmpty={options.every((o) => values.includes(o.value))}
              emptyMessage="Nothing left to filter by."
              // `!atLimit` because Enter is inert at the ceiling: `add` returns
              // without changing `values`, so offering the key would promise
              // nothing.
              noMatchesMessage={
                allowsCustom && !atLimit
                  ? "No match. Press Enter to filter on what you typed."
                  : "No match."
              }
            />
          )}
        >
          {(option: { value: string; label: string }) => (
            <ListBoxItem id={option.value} textValue={option.label}>
              {option.label}
            </ListBoxItem>
          )}
        </ListBox>
      </ComboBox.Popover>
    </ComboBox.Root>
  )
}
