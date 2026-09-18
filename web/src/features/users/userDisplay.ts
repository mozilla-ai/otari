import type { User } from "@/client"

/**
 * How a user id reads on a page that *shows* a person, as opposed to offering
 * one to pick. `userOptionText` in `./userOptions.ts` is the picker's answer to
 * the same question, and the two agree on precedence: the organization roster
 * names the person, the request plane's alias is the fallback, and the id is
 * what is left when nobody named them.
 */
export interface UserDisplay {
  /** What to render: a person's name where one is known, else the id itself. */
  label: string
  /**
   * The raw id, set only when `label` is not already it. A page that has
   * somewhere to put it (a tooltip, a copy control) keeps the id reachable;
   * one that does not, drops it rather than printing `Name (uuid)` in a cell.
   */
  id?: string
}

/**
 * Names the person behind a request-plane user id.
 *
 * `alias` is whatever the gateway was told when the row was written, which the
 * usage API already resolves and ships (`UsageGroupRow.label`,
 * `UsageEntry.user_alias`), so a page rendering those needs no lookup of its
 * own. `memberLabels` is `useMemberAttributionLabels()`, which is who the
 * person is rather than what they were called, so it wins where both exist.
 *
 * An id an operator chose, like `ci-bot`, is already its own name: an alias
 * equal to the id resolves to the id alone, with no second copy to render.
 */
export function userDisplay(
  userId: string,
  alias: string | null | undefined,
  memberLabels: ReadonlyMap<string, string>,
): UserDisplay {
  const name = memberLabels.get(userId) ?? alias
  return name && name !== userId
    ? { label: name, id: userId }
    : { label: userId }
}

/**
 * The aliases the request plane holds, keyed by user id.
 *
 * For the surfaces whose own response carries no label (the budget reset log),
 * where the page already holds the users list and can resolve locally rather
 * than growing a second convention.
 */
export function aliasesByUserId(
  users: User[] | undefined,
): ReadonlyMap<string, string> {
  return new Map(
    (users ?? []).flatMap((user): [string, string][] =>
      user.alias ? [[user.user_id, user.alias]] : [],
    ),
  )
}
