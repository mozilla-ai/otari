import { ComboBox, Input, ListBox, ListBoxItem } from "@heroui/react"
import { type ReactNode, useMemo, useState } from "react"

import type { User } from "@/client"
import { useMemberAttributionLabels } from "@/features/organization/attribution"
import { ControlField } from "@/shared/components/FieldMessages"
import { DismissChip } from "@/shared/components/surface"

interface Option {
  id: string
  label: string
}

const MAX_VISIBLE = 50

// A chip picker of people, for assigning a budget to several at once. Only lists
// named rows (virtual apikey-* shadows are excluded); it never creates one,
// matching the model where a person exists before they are assigned a budget.
//
// The rows are keyed by the id the request plane bills to, which for anyone added
// through the roster is a bare UUID. Showing that is useless to a human, so the
// organization roster is asked what to call them (`attribution_user_id` is the
// join). An id with no member behind it, such as one an operator named directly
// over the API, keeps the id, since that *is* its name.
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

  const labelFor = useMemo(
    () =>
      (user: User): string => {
        const member = memberLabels.get(user.user_id)
        if (member) return member
        return user.alias ? `${user.user_id} (${user.alias})` : user.user_id
      },
    [memberLabels],
  )

  const options = useMemo<Option[]>(
    () =>
      users
        .filter((u) => !u.user_id.startsWith("apikey-"))
        .map((u) => ({ id: u.user_id, label: labelFor(u) })),
    [users, labelFor],
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
            <ListBox items={visible} className="max-h-72 overflow-auto">
              {(option: Option) => (
                <ListBoxItem id={option.id} textValue={option.label}>
                  {option.label}
                </ListBoxItem>
              )}
            </ListBox>
          </ComboBox.Popover>
        </ComboBox.Root>
      )}
    </div>
  )
}
