import type { User } from "@/client"

/** How a person reads in a user picker: what to show, and what to show under it. */
export interface UserOptionText {
  label: string
  /**
   * The id the row submits, when the label does not already carry it. Rendered
   * as the row's muted second line and folded into its accessible name, so two
   * people who share a name stay distinguishable.
   */
  hint?: string
}

/**
 * Names a picker row from the organization roster, falling back to the id.
 *
 * Shared by the two user pickers rather than written in each: they sit in the
 * same forms, and somebody who reads as a name in one and as a UUID in the
 * other reads as two different people (otari-ai#2101).
 *
 * The second line is the person's email, which is what tells two people apart
 * and something an operator recognizes. A member with no address gets none: the
 * id is an internal UUID, so it is shown only where two rows would otherwise
 * read the same (`userOptionTexts`).
 */
export function userOptionText(
  user: User,
  memberLabels: ReadonlyMap<string, string>,
  memberEmails: ReadonlyMap<string, string> = new Map(),
): UserOptionText {
  const member = memberLabels.get(user.user_id)
  if (member) {
    const email = memberEmails.get(user.user_id)
    return email && email !== member
      ? { label: member, hint: email }
      : { label: member }
  }
  // An id an operator chose, like `ci-bot`, is already its own name, so a hint
  // would only repeat the label.
  return {
    label: user.alias ? `${user.user_id} (${user.alias})` : user.user_id,
  }
}

/**
 * `userOptionText` for a whole list, with the id as the hint on any roster row
 * that still shares its label and second line with another: the one case where
 * the UUID is the only thing that distinguishes them.
 */
export function userOptionTexts(
  users: User[],
  memberLabels: ReadonlyMap<string, string>,
  memberEmails: ReadonlyMap<string, string> = new Map(),
): UserOptionText[] {
  const texts = users.map((user) =>
    userOptionText(user, memberLabels, memberEmails),
  )
  const seen = new Map<string, number>()
  for (const t of texts) {
    const key = `${t.label}\u0000${t.hint ?? ""}`
    seen.set(key, (seen.get(key) ?? 0) + 1)
  }
  return texts.map((t, i) => {
    const user = users[i]
    const isShared = (seen.get(`${t.label}\u0000${t.hint ?? ""}`) ?? 0) > 1
    return isShared && memberLabels.has(user.user_id)
      ? { label: t.label, hint: user.user_id }
      : t
  })
}
