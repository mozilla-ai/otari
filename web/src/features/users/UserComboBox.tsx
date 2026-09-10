import type { ReactNode } from "react"
import { useState } from "react"

import type { User } from "@/client"
import {
  ComboBoxField,
  type ComboBoxOption,
} from "@/design-system/forms/ComboBoxField"

interface Option extends ComboBoxOption {
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
      if (member) return { value: u.user_id, label: member, isMember: true }
      return {
        value: u.user_id,
        label: u.alias ? `${u.user_id} (${u.alias})` : u.user_id,
        isMember: false,
      }
    })
    .sort((a, b) => {
      if (a.isMember !== b.isMember) return a.isMember ? -1 : 1
      return a.label.localeCompare(b.label)
    })

  const [text, setText] = useState(value)
  const query = text.trim().toLowerCase()
  const visible = options
    .filter(
      (o) =>
        !query ||
        o.value.toLowerCase().includes(query) ||
        o.label.toLowerCase().includes(query),
    )
    .slice(0, 50)

  // What the operator types is not necessarily a user_id: it may be a label they
  // copied, or an id for a user who does not exist yet. Resolve either form to
  // the canonical id, or the submitted owner would be the label and the keys API
  // would silently create a second user named after it.
  const resolveId = (raw: string): string => {
    const trimmed = raw.trim()
    // An id match outranks a name match, and the order matters because a
    // member's label is a free-form roster name rather than its own id: a roster
    // name can equal another user's `user_id`, members sort to the front, and a
    // single scan matching either field would then bill the key to whichever of
    // the two came first. Only the typed path reaches here. Picking a row reports
    // that row's id, because `ComboBoxField` swallows the display-text echo, so
    // a label shared by two rows cannot resolve to the wrong one.
    const byId = options.find((o) => o.value === trimmed)
    if (byId) return byId.value
    const byName = options.find((o) => o.label === trimmed)
    return byName ? byName.value : trimmed
  }

  const selectedId = resolveId(text)
  const known = options.some((o) => o.value === selectedId)
  const creatingHint =
    selectedId !== "" && !known
      ? (unknownHint ?? (
          <span>
            Creates a new user <code>{selectedId}</code>.
          </span>
        ))
      : (description ?? "Spend and budgets track against this user.")

  return (
    <ComboBoxField
      label={label}
      value={text}
      onChange={(next) => {
        setText(next)
        onChange(resolveId(next))
      }}
      options={visible}
      description={creatingHint}
      placeholder={placeholder}
      allowsCustomValue
      // The whole list on focus, filtered as you type, which is what a
      // pick-from-a-list field wants; nothing here is autofocused.
      menuTrigger="focus"
      // Typing then replaces the shown owner rather than appending to it.
      shouldSelectOnFocus
      // Neither sentence says what typing an id will do, because that differs
      // per caller: the description line below is where `unknownHint` answers
      // it, and promising a creation here would be a 404 on an endpoint that
      // only accepts existing owners.
      isSourceEmpty={options.length === 0}
      emptyMessage="No users to pick from yet."
      noMatchesMessage="No user matches what you typed."
    />
  )
}
