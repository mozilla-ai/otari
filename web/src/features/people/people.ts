/**
 * One row per person on People: the roster joined to what the control plane
 * recorded for them this month. Spend and tokens are Home's People cut, row
 * for row (`useSpend`), so the two pages cannot disagree; sessions, refusals
 * and API keys are counted here from their own reads.
 */

import type {
  ApiKey,
  OrganizationMember,
  TraceSummary,
  UsageGroupRow,
} from "@/client"
import { keyState } from "@/features/keys/keyList"
import { attributionId } from "@/features/organization/attribution"
import { memberLabel, memberRowKey } from "@/features/organization/roles"
import type { SpendRow } from "@/features/spend/useSpend"

export interface PersonRow {
  /** The person's page: their membership id, or their invitation's. */
  id: string
  member: OrganizationMember
  /** The id their requests, sessions and keys are recorded under. */
  userId: string | null
  name: string
  email: string
  /** The workspaces they are placed in. */
  teams: string[]
  role: string
  isInvited: boolean
  spend: number
  /** Billed tokens, cache reads included, as Home counts them. */
  tokens: number
  /** Their row in Home's People cut, which their top model is read for. */
  spendRow: SpendRow | undefined
  sessions: number
  /** Requests the gateway refused them this month (status 403). */
  refusals: number
  /** Their active API keys, or undefined where the caller cannot read them. */
  apiKeys: number | undefined
  /**
   * In no workspace the caller can read. Usage and sessions reach a member or
   * viewer only for their own workspaces, so this person's zeros would be the
   * caller's blind spot rather than the person's month.
   */
  isOutsideView: boolean
  /**
   * The reads whose figures on this row are unknown rather than zero, because
   * the read failed. A column shows its own unknown and keeps the rest.
   */
  unread: UnreadReads
}

/** The reads a row's columns stand on, by the column they feed. */
export interface UnreadReads {
  /** Spend, tokens, requests and the top model. */
  spend: boolean
  sessions: boolean
  refusals: boolean
  /** The API key count. */
  keys: boolean
}

export const ALL_READ: UnreadReads = {
  spend: false,
  sessions: false,
  refusals: false,
  keys: false,
}

export interface MonthRecord {
  /** The month's clock: a key past its expiry is not active at it. */
  now: number
  /** The reads that failed; anything absent was read. */
  unread?: Partial<UnreadReads>
  /** Home's People cut: `useSpend(month).people`. */
  spendRows: readonly SpendRow[]
  refusalsByUser: readonly UsageGroupRow[]
  sessions: readonly TraceSummary[]
  /**
   * Sessions counted by person without listing them, where only some people's
   * are wanted: a person's page. Wins over `sessions` for the people it holds.
   */
  sessionCounts?: ReadonlyMap<string, number>
  /** Every key the caller may read. */
  keys: readonly ApiKey[]
  /**
   * The memberships whose keys the caller can read; undefined for everyone's.
   * Only the deployment's operator lists every key: anyone else lists their
   * own, so another person's count would read as zero when it is unknown.
   */
  keyReaderMemberIds?: ReadonlySet<string>
  /** The organization's workspaces: sessions anywhere else are not counted. */
  workspaceIds?: ReadonlySet<string>
  /** The workspaces the caller's reads cover; undefined for all of them. */
  visibleWorkspaceIds?: ReadonlySet<string>
  /**
   * List only the people placed in this workspace; everyone when absent. Their
   * spend, sessions and refusals are read for the same workspace (`usePeople`);
   * their key count is every active key they own.
   */
  workspaceId?: string
}

/**
 * The spend limit on a person in each workspace, as the roster carries it to
 * anyone in the organization: the amount only. Its period, its name and how
 * much of it is gone are on the ceiling itself, which only owners and admins
 * may read. Workspaces with no limit on them are left out.
 */
export function rosterLimits(
  member: Pick<OrganizationMember, "workspaces">,
): { workspaceId: string; team: string; amount: number }[] {
  return (member.workspaces ?? [])
    .flatMap((place) =>
      // A $0 limit is a limit: it refuses every request.
      typeof place.ceiling?.max_budget === "number"
        ? [
            {
              workspaceId: place.workspace_id,
              team: place.workspace_name,
              amount: place.ceiling.max_budget,
            },
          ]
        : [],
    )
    .sort((a, b) => a.team.localeCompare(b.team))
}

/** The active keys a member owns (not disabled, not expired), by either id a key can be issued under. */
export function activeKeysOf(
  keys: readonly ApiKey[],
  member: OrganizationMember,
  now: number,
): ApiKey[] {
  const ids = new Set(
    [attributionId(member), member.user_id].filter(
      (id): id is string => typeof id === "string",
    ),
  )
  return keys.filter(
    (key) =>
      keyState(key, now) === "active" &&
      key.user_id !== null &&
      ids.has(key.user_id),
  )
}

export function personRows(
  members: readonly OrganizationMember[],
  record: MonthRecord,
): PersonRow[] {
  const spendRows = new Map(record.spendRows.map((row) => [row.id, row]))
  const refused = new Map(
    record.refusalsByUser.map((row) => [row.key, row.requests]),
  )
  const sessionCount = new Map<string, number>()
  for (const session of record.sessions) {
    if (!session.user_id) continue
    if (record.workspaceIds && !record.workspaceIds.has(session.workspace_id)) {
      continue
    }
    sessionCount.set(
      session.user_id,
      (sessionCount.get(session.user_id) ?? 0) + 1,
    )
  }
  return members
    .filter(
      (member) => member.status === "active" || member.status === "invited",
    )
    .filter(
      (member) =>
        record.workspaceId === undefined ||
        (member.workspaces ?? []).some(
          (place) => place.workspace_id === record.workspaceId,
        ),
    )
    .map((member) => {
      const userId = attributionId(member)
      const visible = record.visibleWorkspaceIds
      const isOutsideView =
        visible !== undefined &&
        !(member.workspaces ?? []).some((place) =>
          visible.has(place.workspace_id),
        )
      const has = userId !== null && !isOutsideView
      // `useSpend` keys a member's row by their membership.
      const spendRow =
        has && member.organization_member_id
          ? spendRows.get(`person:${member.organization_member_id}`)
          : undefined
      const readers = record.keyReaderMemberIds
      const canReadKeys =
        readers === undefined ||
        readers.has(member.organization_member_id ?? "")
      return {
        id: memberRowKey(member),
        member,
        userId,
        name: memberLabel(member),
        email: member.email ?? "",
        teams: (member.workspaces ?? []).map((place) => place.workspace_name),
        role: member.role,
        isInvited: member.status === "invited",
        spend: spendRow?.cost ?? 0,
        tokens: spendRow?.tokens ?? 0,
        spendRow,
        sessions: has
          ? (record.sessionCounts?.get(userId) ?? sessionCount.get(userId) ?? 0)
          : 0,
        refusals: has ? (refused.get(userId) ?? 0) : 0,
        apiKeys: canReadKeys
          ? activeKeysOf(record.keys, member, record.now).length
          : undefined,
        isOutsideView,
        unread: {
          ...ALL_READ,
          ...record.unread,
          spend: Boolean(record.unread?.spend || spendRow?.isFolded),
        },
      }
    })
    .sort(
      (a, b) =>
        Number(a.isInvited) - Number(b.isInvited) ||
        Number(a.isOutsideView) - Number(b.isOutsideView) ||
        b.spend - a.spend ||
        a.name.localeCompare(b.name),
    )
}
