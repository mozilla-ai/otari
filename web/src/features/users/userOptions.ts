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
 */
export function userOptionText(
  user: User,
  memberLabels: ReadonlyMap<string, string>,
): UserOptionText {
  const member = memberLabels.get(user.user_id)
  if (member) return { label: member, hint: user.user_id }
  // An id an operator chose, like `ci-bot`, is already its own name, so a hint
  // would only repeat the label.
  return {
    label: user.alias ? `${user.user_id} (${user.alias})` : user.user_id,
  }
}
