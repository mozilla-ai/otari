import type { ReactNode } from "react"

import type { User } from "@/client"
import {
  ComboBoxField,
  type ComboBoxOption,
} from "@/design-system/forms/ComboBoxField"

import { userOptionText } from "./userOptions"

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
  onQueryChange,
  description,
  label = "Owner",
  placeholder = "Pick a user, or type a new id…",
  unknownHint,
}: {
  value: string
  onChange: (userId: string) => void
  users: User[]
  /**
   * Where to report what is being typed. The field owns the input's text; the
   * page owns the term, because the page is what fetches. A field that filtered
   * `users` itself would offer the matches out of whatever page had been
   * fetched (otari#1380).
   */
  onQueryChange: (query: string) => void
  description?: ReactNode
  label?: ReactNode
  placeholder?: string
  // What to say when the typed id is not an existing user. Defaults to the
  // keys-page truth (that endpoint creates the user); callers whose endpoint
  // rejects an unknown id must override it rather than promise a creation that
  // will 404.
  unknownHint?: ReactNode
}) {
  // The id stays the value submitted whatever the row reads as, and rides along
  // as the hint so it is still what a search can match.
  const options: ComboBoxOption[] = users
    .filter((user) => !user.user_id.startsWith("apikey-"))
    .map((user) => ({
      value: user.user_id,
      ...userOptionText(user),
    }))
    .sort((a, b) => {
      // A roster-named row is the one carrying a hint, and it sorts to the
      // front: a member is who someone means when issuing a key, where a
      // hand-made id like `ci-bot` is a tool.
      if (Boolean(a.hint) !== Boolean(b.hint)) return a.hint ? -1 : 1
      return a.label.localeCompare(b.label)
    })

  // Rendered as handed over: the narrowing is the server's, and filtering here
  // again would hide a match that arrived while the field's text moved on.
  const visible = options.slice(0, 50)

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
      onQueryChange={onQueryChange}
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
