/**
 * The API keys page as rules: a key's state, which keys the State filter lists,
 * the order they come in, and what the row menu offers the caller.
 *
 * The menu follows the server, not the page: a write the server would refuse
 * is never offered. The rules are `organization_keys.py` (`_load_own_writable_key`,
 * `_federation_rule_key`) and `require_workspace_contributor_access` in
 * `organization_service.py`, in otari.
 */

import type { ApiKey } from "@/client"
import {
  isContributorRole,
  isManagementRole,
} from "@/features/organization/roles"

/** What a key is called on screen, named or not. */
export function keyLabel(key: { key_name: string | null }): string {
  return key.key_name || "Unnamed key"
}

export type KeyState = "active" | "disabled" | "expired"

/**
 * Disabled first: a switched-off key is refused whatever its expiry says.
 */
export function keyState(key: ApiKey, now: number): KeyState {
  if (!key.is_active) return "disabled"
  if (key.expires_at) {
    const ends = Date.parse(key.expires_at)
    if (!Number.isNaN(ends) && ends < now) return "expired"
  }
  return "active"
}

export const STATE_WORDS: Record<KeyState, string> = {
  active: "Active",
  disabled: "Disabled",
  expired: "Expired",
}

type StateFilter = KeyState | "all"

export const STATE_FILTERS: { value: StateFilter; label: string }[] = [
  { value: "active", label: "Active" },
  { value: "disabled", label: "Disabled" },
  { value: "expired", label: "Expired" },
  { value: "all", label: "Every state" },
]

export function parseStateFilter(value: string): StateFilter {
  return (
    STATE_FILTERS.find((option) => option.value === value)?.value ?? "active"
  )
}

const STATE_ORDER: Record<KeyState, number> = {
  active: 0,
  expired: 1,
  disabled: 2,
}

export interface KeyRow {
  key: ApiKey
  state: KeyState
  /**
   * Spent this month, in dollars; 0 when the usage read has no row for it, and
   * null until that read answers or when it folded this key into its Other row.
   */
  spend: number | null
  tokens: number | null
}

/**
 * A key's row, with this month's usage once it has been read: the summary's
 * rows by key. Its Other row has no key and comes after every real row, the
 * one with none included, so it is what the summary holds under none.
 */
export function keyRow(
  key: ApiKey,
  usage: {
    get(
      id: string | null,
    ): { cost: number; tokens: number; is_other?: boolean } | undefined
  } | null,
  now: number,
): KeyRow {
  const used = usage?.get(key.id)
  const isFolded = !used && usage?.get(null)?.is_other === true
  const isKnown = usage !== null && !isFolded
  return {
    key,
    state: keyState(key, now),
    spend: isKnown ? (used?.cost ?? 0) : null,
    tokens: isKnown ? (used?.tokens ?? 0) : null,
  }
}

/**
 * Active, then Expired, then Disabled; inside each the biggest spender first,
 * a key whose spend is unknown after every known one, then by name.
 */
export function compareRows(a: KeyRow, b: KeyRow): number {
  const bySpend =
    a.spend === null || b.spend === null
      ? Number(a.spend === null) - Number(b.spend === null)
      : b.spend - a.spend
  return (
    STATE_ORDER[a.state] - STATE_ORDER[b.state] ||
    bySpend ||
    (a.key.key_name ?? "").localeCompare(b.key.key_name ?? "")
  )
}

export function rowsInState(
  rows: readonly KeyRow[],
  filter: StateFilter,
): KeyRow[] {
  return filter === "all" ? [...rows] : rows.filter((r) => r.state === filter)
}

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`

/** The figure at the right end of the toolbar. */
export function countLine(
  rows: readonly KeyRow[],
  filter: StateFilter,
): string {
  const active = rows.filter((r) => r.state === "active").length
  if (filter === "active") return `${active} active of ${rows.length}`
  if (filter === "all") return `${plural(rows.length, "key")}, ${active} active`
  return `${rows.filter((r) => r.state === filter).length} ${filter}`
}

/**
 * Whether the list holds anyone else's key. Only the deployment operator is
 * listed other people's keys; an owner or admin is listed the organization's
 * service keys as well, which the Key cell marks, so they do not count.
 */
export function listsOtherPeoplesKeys(
  keys: readonly ApiKey[],
  me: string | undefined,
): boolean {
  return keys.some(
    (key) => !key.is_organization_service_key && key.user_id !== me,
  )
}

export interface Caller {
  /** The deployment operator acts on every key, through `/api/v1/keys`. */
  isOperator: boolean
  /** The organization role. */
  role: string | undefined
  /** The caller's role in each workspace they are an active member of. */
  workspaceRoles: ReadonlyMap<string, string>
}

/**
 * What the server lets the caller do to one key it lists to them.
 *
 * - `write`: change, regenerate, switch on or off, delete.
 * - `revoke`: switch it off or delete it, nothing else. A member removed from
 *   the key's workspace keeps that much over their own key.
 * - `read`: nothing. A viewer of the organization, or of the key's workspace,
 *   changes no key, their own included: the stricter of the two roles wins.
 */
type KeyAccess = "write" | "revoke" | "read"

export function keyAccess(key: ApiKey, caller: Caller): KeyAccess {
  if (caller.isOperator || isManagementRole(caller.role)) return "write"
  if (!isContributorRole(caller.role)) return "read"
  const workspaceRole = caller.workspaceRoles.get(key.workspace_id)
  if (workspaceRole === undefined) return "revoke"
  return isContributorRole(workspaceRole) ? "write" : "read"
}

/**
 * Adding a federation rule takes an owner or admin of the organization, or the
 * operator, and a deployment that lets organizations manage them at all.
 */
export function canAddWorkloadIdentity(
  caller: Caller,
  federationMemberRules: boolean,
): boolean {
  return (
    federationMemberRules &&
    (caller.isOperator || isManagementRole(caller.role))
  )
}

export type KeyMenuId =
  | "edit"
  | "disable"
  | "enable"
  | "end-users"
  | "workload-identity"
  | "regenerate"
  | "sessions"
  | "delete"

export interface KeyMenuItem {
  id: KeyMenuId
  label: string
  hint: string
  isDisabled?: boolean
  isDestructive?: boolean
}

/**
 * The row menu for one key, in order. An item the caller cannot use is left
 * out rather than shown disabled, except two: Delete while the key is on,
 * and Enable on a key an operator switched off, which only an operator may
 * switch back on.
 */
export function keyMenu({
  key,
  access,
  isOperator,
  canFederate,
  isPending = false,
}: {
  key: ApiKey
  access: KeyAccess
  isOperator: boolean
  canFederate: boolean
  isPending?: boolean
}): KeyMenuItem[] {
  const items: KeyMenuItem[] = []
  const writes = access === "write"
  const revokes = access === "write" || access === "revoke"
  if (writes) {
    items.push({
      id: "edit",
      label: "Edit…",
      hint: "Name, expiry, models and tools",
    })
  }
  if (revokes && key.is_active) {
    items.push({
      id: "disable",
      label: "Disable",
      hint: "Its requests are refused until you enable it again",
      isDisabled: isPending,
    })
  }
  if (writes && !key.is_active) {
    // Only an operator clears a reason, and only one the server says is theirs.
    const isLocked =
      key.disabled_reason != null && !(isOperator && key.operator_can_enable)
    items.push({
      id: "enable",
      label: "Enable",
      hint: !isLocked
        ? "Requests are accepted again"
        : key.disabled_reason === "revoked"
          ? "Revoked permanently"
          : "Disabled by an operator",
      isDisabled: isLocked || isPending,
    })
  }
  if (key.is_service_key) {
    items.push({
      id: "end-users",
      label: "End users…",
      hint: "Who it has acted for, and their spend limits",
    })
  }
  if (writes && canFederate) {
    items.push({
      id: "workload-identity",
      label: "Workload identity…",
      hint: "Use it without the secret",
    })
  }
  if (writes) {
    items.push({
      id: "regenerate",
      label: "Regenerate…",
      hint: "New secret; the old one stops working",
      isDisabled: isPending,
    })
  }
  items.push({
    id: "sessions",
    label: "See its sessions",
    hint: "What ran on it this month",
  })
  if (revokes) {
    items.push({
      id: "delete",
      label: "Delete…",
      hint: key.is_active ? "Disable it first" : "Cannot be undone",
      isDisabled: key.is_active || isPending,
      isDestructive: true,
    })
  }
  return items
}
