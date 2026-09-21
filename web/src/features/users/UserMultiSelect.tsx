import type { ReactNode } from "react"

import type { User } from "@/client"
import { MultiSelect } from "@/design-system/forms/MultiSelect"

import { userOptionText } from "./userOptions"

// A picker of people, for assigning a budget to several at once. Only lists
// named rows (virtual apikey-* shadows are excluded); it never creates one,
// matching the model where a person exists before they are assigned a budget.
//
// The rows are keyed by the id the request plane bills to, which for anyone added
// through the roster is a bare UUID. Showing that is useless to a human, so the
// organization roster is asked what to call them (`attribution_user_id` is the
// join). An id with no member behind it, such as one an operator named directly
// over the API, keeps the id, since that *is* its name.
//
// Everything about how the picking behaves is `forms/MultiSelect`'s now. What is
// left here is the one thing a design-system component cannot know: which rows
// are people, and what a person is called.
export function UserMultiSelect({
  value,
  onChange,
  users,
  onQueryChange,
  label,
  description,
}: {
  value: string[]
  onChange: (next: string[]) => void
  users: User[]
  /**
   * Where to report what is being typed. The page fetches the matches, so the
   * server matches over everyone rather than over the page already fetched
   * (otari#1380).
   */
  onQueryChange: (query: string) => void
  label: string
  description?: ReactNode
}) {
  const options = users
    .filter((user) => !user.user_id.startsWith("apikey-"))
    .map((user) => ({
      id: user.user_id,
      ...userOptionText(user),
    }))

  return (
    <MultiSelect
      label={label}
      description={description}
      options={options}
      value={value}
      onChange={onChange}
      onQueryChange={onQueryChange}
      searchPlaceholder="Search people…"
      countNoun={{ one: "person assigned", other: "people assigned" }}
      emptyMessage="Nobody to assign yet. Add people under Members & roles and they can be assigned here."
      noMatchesMessage="Nobody matches what you typed."
    />
  )
}
