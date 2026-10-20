import { type ReactNode, useState } from "react"

import type { User } from "@/client"
import {
  ComboBoxField,
  type ComboBoxOption,
} from "@/design-system/forms/ComboBoxField"
import {
  useMemberAttributionEmails,
  useMemberAttributionLabels,
} from "@/features/organization/attribution"

import { isVirtualUser, userOptionTexts } from "./userOptions"

// A required "owner" picker for a new API key: choose an existing user or type a
// new id to create one (the keys API creates a named user for any id it does not
// know). This is the dashboard's user-first gate; it never mints an anonymous
// virtual user the way an omitted id at the API would. Virtual users are left out
// of the options: you attach keys to people/teams you name, not to key-shadows.
//
// `value` is the owner id, never the name shown for it. Picking a row is how an
// existing person is chosen; typed text is an id of its own, so typing somebody's
// display name names a new user rather than resolving to their UUID.
export function UserComboBox({
  value,
  onChange,
  users,
  description,
  label = "Owner",
  placeholder = "Pick a user, or type a new id…",
  unknownHint,
  isRequired,
}: {
  value: string
  onChange: (userId: string) => void
  users: User[]
  description?: ReactNode
  label?: ReactNode
  placeholder?: string
  // What to say when the typed id is not an existing user. Defaults to the
  // keys-page truth (that endpoint creates the user); callers whose endpoint
  // rejects an unknown id must override it rather than promise a creation that
  // will 404.
  unknownHint?: ReactNode
  isRequired?: boolean
}) {
  // Read here rather than taken as a prop, so no call site can forget it and
  // leave a person reading as the UUID their identity was minted under. The
  // read is gated on the `organizations` surface and shares its query key with
  // every other reader of the roster.
  const memberLabels = useMemberAttributionLabels()
  const memberEmails = useMemberAttributionEmails()

  // The id stays the value submitted whatever the row reads as, and a search
  // still matches it. The second line is the email, or the id only where two
  // rows would otherwise be indistinguishable.
  const owners = users.filter((user) => !isVirtualUser(user.user_id))
  const texts = userOptionTexts(owners, memberLabels, memberEmails)
  const options: ComboBoxOption[] = owners
    .map((user, i) => ({ value: user.user_id, ...texts[i] }))
    .sort((a, b) => {
      // A roster member sorts to the front: a member is who someone means when
      // issuing a key, where a hand-made id like `ci-bot` is a tool.
      const aMember = memberLabels.has(a.value)
      if (aMember !== memberLabels.has(b.value)) return aMember ? -1 : 1
      return a.label.localeCompare(b.label)
    })

  // What is being searched for, reported by the field because it owns the
  // input's text. Not the value: an id an operator types is a value, and a name
  // they type is only ever a search.
  const [query, setQuery] = useState("")
  const q = query.trim().toLowerCase()
  const visible = options
    .filter(
      (option) =>
        !q ||
        option.value.toLowerCase().includes(q) ||
        option.label.toLowerCase().includes(q) ||
        option.hint?.toLowerCase().includes(q),
    )
    .slice(0, 50)

  const ownerId = value.trim()
  const isKnownOwner = options.some((option) => option.value === ownerId)
  const creatingHint =
    ownerId !== "" && !isKnownOwner
      ? (unknownHint ?? (
          <span>
            Creates a new user <code>{ownerId}</code>.
          </span>
        ))
      : (description ?? "Spend and budgets track against this user.")

  return (
    <ComboBoxField
      label={label}
      value={value}
      // Trimmed, because a pasted id often carries a space and every caller
      // submits this as an owner id.
      onChange={(next) => onChange(next.trim())}
      onQueryChange={setQuery}
      options={visible}
      description={creatingHint}
      isRequired={isRequired}
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
