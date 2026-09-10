import { ComboBox, Input, ListBox, ListBoxItem } from "@heroui/react"
import { type ReactNode, useMemo, useState } from "react"

import type { User } from "@/client"
import { ComboBoxEmpty } from "@/design-system/forms/ComboBoxEmpty"
import { comboBoxOptionText } from "@/design-system/forms/ComboBoxField"
import { ControlField } from "@/design-system/forms/FieldMessages"
import { DismissChip } from "@/design-system/indicators/DismissChip"
import { useMemberAttributionLabels } from "@/features/organization/attribution"

import { type UserOptionText, userOptionText } from "./userOptions"

interface Option extends UserOptionText {
  id: string
}

const MAX_VISIBLE = 50

// A chip picker of people, for assigning a budget to several at once. Only lists
// named rows (virtual apikey-* shadows are excluded); it never creates one,
// matching the model where a person exists before they are assigned a budget.
//
// Rows are keyed by the id the request plane bills to, which for anyone added
// through the roster is a bare UUID (`attribution_user_id` is the join).
// `userOptionText` decides what that reads as, so this picker and the owner
// picker name a person the same way.
export function UserMultiSelect({
  value,
  onChange,
  users,
  label,
  description,
}: {
  value: string[]
  onChange: (next: string[]) => void
  users: User[]
  label: string
  description?: ReactNode
}) {
  const [query, setQuery] = useState("")
  const memberLabels = useMemberAttributionLabels()

  const options = useMemo<Option[]>(
    () =>
      users
        .filter((u) => !u.user_id.startsWith("apikey-"))
        .map((u) => ({ id: u.user_id, ...userOptionText(u, memberLabels) })),
    [users, memberLabels],
  )
  const labelById = useMemo(
    () => new Map(options.map((option) => [option.id, option.label])),
    [options],
  )

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase()
    return options
      .filter((o) => !value.includes(o.id))
      .filter(
        (o) =>
          !q ||
          o.id.toLowerCase().includes(q) ||
          o.label.toLowerCase().includes(q),
      )
      .slice(0, MAX_VISIBLE)
  }, [options, value, query])

  const add = (id: string) => {
    if (!value.includes(id)) onChange([...value, id])
    setQuery("")
  }
  const remove = (id: string) => onChange(value.filter((v) => v !== id))

  return (
    <div className="flex flex-col gap-2">
      <ControlField label={label} description={description} />
      {value.length > 0 ? (
        <div className="flex flex-wrap gap-1.5">
          {value.map((id) => (
            <DismissChip
              key={id}
              value={labelById.get(id) ?? id}
              onDismiss={() => remove(id)}
              dismissLabel={`Remove ${labelById.get(id) ?? id}`}
            />
          ))}
        </div>
      ) : null}
      {options.length === 0 ? (
        <span className="text-caption">
          Nobody to assign yet. Add people under Members &amp; roles, or issue a
          key, and they can be assigned here.
        </span>
      ) : (
        <ComboBox.Root
          allowsEmptyCollection
          menuTrigger="input"
          inputValue={query}
          onInputChange={setQuery}
          selectedKey={null}
          onSelectionChange={(key) => {
            if (key != null) add(String(key))
          }}
          className="flex flex-col gap-1"
        >
          <ComboBox.InputGroup>
            <Input
              aria-label="Add a person"
              placeholder="Search people…"
              autoComplete="off"
            />
            <ComboBox.Trigger />
          </ComboBox.InputGroup>
          <ComboBox.Popover>
            <ListBox
              items={visible}
              className="max-h-72 overflow-auto"
              renderEmptyState={() => (
                <ComboBoxEmpty
                  isSourceEmpty={options.every((o) => value.includes(o.id))}
                  emptyMessage="Everybody here is already assigned."
                  noMatchesMessage="Nobody matches what you typed."
                />
              )}
            >
              {(option: Option) => (
                <ListBoxItem
                  id={option.id}
                  textValue={comboBoxOptionText(option)}
                  // Spelled out for the reason `ComboBoxField` spells it out: a
                  // two-line row's computed name runs the hint onto the label
                  // with no separator, and the hint is what tells two people of
                  // one name apart.
                  aria-label={
                    option.hint ? comboBoxOptionText(option) : undefined
                  }
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
        </ComboBox.Root>
      )}
    </div>
  )
}
