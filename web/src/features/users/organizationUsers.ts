import type { ApiKey, User } from "@/client"

/**
 * The request-plane users one organization can name, out of the deployment's whole list.
 *
 * `GET /users` is deployment-wide: it serves every spend identity in the
 * process, which on a deployment holding mutually-untrusting tenants is every
 * tenant's people. A key is minted inside one organization (`POST /keys`
 * resolves the caller's active one and refuses a workspace outside it), so an
 * owner picker fed that list offered another organization's people as owners
 * (otari-ai#2108).
 *
 * A `users` row carries no organization, so this derives the scope from the two
 * joins that do exist: the roster names its members' attribution rows, and a key
 * names both its owner and the workspace, hence the organization, it lives in.
 *
 * An id in neither is left out rather than shown. It belongs to another
 * organization, or to nobody who has used this one yet, and either way a picker
 * that still accepts a typed id can reach it.
 */
export function organizationUsers(
  users: User[],
  memberLabels: ReadonlyMap<string, string>,
  organizationKeys: readonly Pick<ApiKey, "user_id">[],
): User[] {
  const owners = new Set(
    organizationKeys.flatMap((key) => (key.user_id ? [key.user_id] : [])),
  )
  return users.filter(
    (user) => memberLabels.has(user.user_id) || owners.has(user.user_id),
  )
}
