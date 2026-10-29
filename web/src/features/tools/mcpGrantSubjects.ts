import type { ApiKey, WorkspaceMcpServerGrant } from "@/client"

/** One key a grant can name, as the picker lists it. */
export interface GrantKey {
  id: string
  name: string
  isActive: boolean
}

/** A user and the keys of theirs the page can see, one group of the picker. */
export interface GrantUser {
  userId: string
  label: string
  keys: GrantKey[]
}

/** How a key reads when it has no name of its own. */
export function keyName(name: string | null | undefined, id: string): string {
  return name || `Key ${id.slice(0, 8)}`
}

/** How a user id reads: their name where the roster knows it, else the id. */
export function userLabel(
  userId: string,
  labels: ReadonlyMap<string, string>,
): string {
  return labels.get(userId) ?? userId
}

/**
 * The picker's groups: each user with the keys of theirs listed, then the
 * users who own no listed key, sorted by name so a user is found where the
 * eye expects them.
 *
 * Keys come from the keys surface, which shows an operator every key and
 * anyone else their own; `userIdsInWorkspace` adds the workspace's other members, who
 * can still be granted as users. A key with no owner is left out: there is
 * nobody to group it under, and a user grant could never cover it.
 */
export function grantUsers(
  keys: readonly ApiKey[],
  userIdsInWorkspace: readonly string[],
  labels: ReadonlyMap<string, string>,
): GrantUser[] {
  const groups = new Map<string, GrantUser>()
  const groupFor = (userId: string) => {
    let group = groups.get(userId)
    if (!group) {
      group = { userId, label: userLabel(userId, labels), keys: [] }
      groups.set(userId, group)
    }
    return group
  }
  for (const key of keys) {
    if (!key.user_id) continue
    groupFor(key.user_id).keys.push({
      id: key.id,
      name: keyName(key.key_name, key.id),
      isActive: key.is_active,
    })
  }
  for (const userId of userIdsInWorkspace) groupFor(userId)
  return [...groups.values()].sort((a, b) => a.label.localeCompare(b.label))
}

/** Whether a group, or any key in it, matches what was typed in the search. */
export function filterUsers(
  users: readonly GrantUser[],
  query: string,
): GrantUser[] {
  const needle = query.trim().toLowerCase()
  if (needle === "") return [...users]
  return users.flatMap((user) => {
    if (user.label.toLowerCase().includes(needle)) return [user]
    const keys = user.keys.filter((key) =>
      key.name.toLowerCase().includes(needle),
    )
    return keys.length > 0 ? [{ ...user, keys }] : []
  })
}

/** Who one grant is for, as the ACL table names it. */
export function grantSubject(
  grant: WorkspaceMcpServerGrant,
  labels: ReadonlyMap<string, string>,
): { name: string; detail: string } {
  const owner = grant.user_id ? userLabel(grant.user_id, labels) : ""
  if (grant.subject === "user") {
    return { name: owner || "Unknown user", detail: "user, all their keys" }
  }
  return {
    name: keyName(grant.key_name, grant.api_key_id ?? grant.id),
    detail: owner ? `key of ${owner}` : "key",
  }
}
