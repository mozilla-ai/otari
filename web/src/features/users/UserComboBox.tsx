import {
  ComboBox,
  Description,
  Input,
  Label,
  ListBox,
  ListBoxItem,
} from "@heroui/react"
import type { ReactNode } from "react"
import { useState } from "react"

import type { User } from "@/client"

interface Option {
  id: string
  name: string
  isMember: boolean
}

// A required "owner" picker for a new API key: choose an existing user or type a
// new id to create one (the keys API creates a named user for any id it does not
// know). This is the dashboard's user-first gate; it never mints an anonymous
// virtual user the way an omitted id at the API would. Virtual users are left out
// of the options: you attach keys to people/teams you name, not to key-shadows.
export function UserComboBox({
  value,
  onChange,
  users,
  description,
  label = "Owner",
  placeholder = "Pick a user, or type a new id…",
  unknownHint,
  memberLabels,
}: {
  value: string
  onChange: (userId: string) => void
  users: User[]
  description?: ReactNode
  label?: ReactNode
  placeholder?: string
  // Names the organization members among these users, keyed by the owner id
  // they bill through. Without it a member reads as the bare UUID their
  // identity was minted under, which nobody can pick from a list.
  memberLabels?: ReadonlyMap<string, string>
  // What to say when the typed id is not an existing user. Defaults to the
  // keys-page truth (that endpoint creates the user); callers whose endpoint
  // rejects an unknown id must override it rather than promise a creation that
  // will 404.
  unknownHint?: ReactNode
}) {
  // A member is named by the roster and sorted to the front: those are the
  // owners someone means when issuing a key, and their raw id is a UUID that
  // reads as noise next to a hand-made one like `ci-bot`. The id stays the
  // value submitted either way; only the label changes.
  const options: Option[] = users
    .filter((u) => !u.user_id.startsWith("apikey-"))
    .map((u) => {
      const member = memberLabels?.get(u.user_id)
      if (member) return { id: u.user_id, name: member, isMember: true }
      return {
        id: u.user_id,
        name: u.alias ? `${u.user_id} (${u.alias})` : u.user_id,
        isMember: false,
      }
    })
    .sort((a, b) => {
      if (a.isMember !== b.isMember) return a.isMember ? -1 : 1
      return a.name.localeCompare(b.name)
    })

  const [text, setText] = useState(value)
  const query = text.trim().toLowerCase()
  const visible = options
    .filter(
      (o) =>
        !query ||
        o.id.toLowerCase().includes(query) ||
        o.name.toLowerCase().includes(query),
    )
    .slice(0, 50)

  // What the input holds is not necessarily the user_id: when an option is picked
  // the ComboBox writes that option's display text back into the input, which
  // re-fires onInputChange. Resolve either form (raw id, or "id (alias)" label) to
  // the canonical id, or the submitted owner would be the label and the keys API
  // would silently create a second user named after it.
  const resolveId = (raw: string): string => {
    const trimmed = raw.trim()
    // An id match outranks a name match, and the order matters now that a
    // member's label is a free-form roster name rather than its own id: a roster
    // name can equal another user's `user_id`, members sort to the front, and a
    // single scan matching either field would then bill the key to whichever of
    // the two came first. Only the typed path is affected (picking an option
    // carries the item's id), which is exactly the path that takes an id.
    const byId = options.find((o) => o.id === trimmed)
    if (byId) return byId.id
    const byName = options.find((o) => o.name === trimmed)
    return byName ? byName.id : trimmed
  }

  const selectedId = resolveId(text)
  const known = options.some((o) => o.id === selectedId)
  const creatingHint =
    selectedId !== "" && !known
      ? (unknownHint ?? (
          <span>
            Creates a new user <code>{selectedId}</code>.
          </span>
        ))
      : (description ?? "Spend and budgets track against this user.")

  return (
    <ComboBox.Root
      allowsCustomValue
      allowsEmptyCollection
      menuTrigger="focus"
      inputValue={text}
      onInputChange={(next) => {
        setText(next)
        onChange(resolveId(next))
      }}
      onSelectionChange={(key) => {
        if (key != null) {
          const id = String(key)
          setText(id)
          onChange(id)
        }
      }}
      // Cap the width so the field and its dropdown trigger stay within easy
      // reach instead of stretching across a wide form.
      className="flex max-w-md flex-col gap-1"
    >
      <Label className="text-body">{label}</Label>
      <ComboBox.InputGroup>
        {/* Not a credential field: keep password managers out, and select on focus
            so typing replaces the current value rather than appending. */}
        <Input
          placeholder={placeholder}
          autoComplete="off"
          data-1p-ignore
          data-lpignore="true"
          onFocus={(event) => event.currentTarget.select()}
        />
        <ComboBox.Trigger />
      </ComboBox.InputGroup>
      <ComboBox.Popover>
        <ListBox items={visible} className="max-h-72 overflow-auto">
          {(option: Option) => (
            <ListBoxItem id={option.id} textValue={option.name}>
              {option.name}
            </ListBoxItem>
          )}
        </ListBox>
      </ComboBox.Popover>
      <Description className="text-caption">{creatingHint}</Description>
    </ComboBox.Root>
  )
}
