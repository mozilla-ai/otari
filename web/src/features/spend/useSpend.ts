import type {
  ApiKey,
  OrganizationMember,
  OrganizationSpendCeiling,
  UsageGroupRow,
  UsageSeriesPoint,
  UsageTotals,
} from "@/client"
import { keyState } from "@/features/keys/keyList"
import { useIsOrganizationManager } from "@/features/organization/access"
import {
  type AppliedLimit,
  type LimitScope,
  type LimitTarget,
  limitsOn,
  limitsOnPerson,
  NO_LIMITS,
  type RowLimits,
  resetEndedPeriods,
  UNKNOWN_LIMITS,
} from "@/features/spend/limits"
import { keyName, memberName } from "@/features/spend/names"
import { type SpendWindow, usageRange } from "@/features/spend/period"
import { groupModels, OTHER } from "@/features/usage/modelGroups"
import { billedTokenTotal } from "@/features/usage/usageTotals"
import { useKeys } from "@/shared/api/apiKeys"
import { useAllSpendCeilings } from "@/shared/api/budgets"
import {
  useOrganizationContext,
  useOrganizationMembers,
} from "@/shared/api/organizations"
import {
  NO_BREAKDOWNS,
  useUsageSummary,
  useWorkspaceSpendSummaries,
} from "@/shared/api/usage"
import { useWorkspaces } from "@/shared/api/workspaces"
import { providerDisplayName } from "@/shared/helpers/providers"

export interface SpendRow {
  id: string
  /**
   * The id Sessions filters on (Requests, for a provider), or null for a row
   * that is not one spender.
   */
  filterKey: string | null
  /**
   * A model's row: every id it was requested as, which a Sessions filter
   * names together (`filterValues`). `filterKey` is the busiest of them.
   */
  spellings?: string[]
  name: string
  detail?: string
  cost: number
  requests: number
  tokens: number
  limits: RowLimits
  /** What a limit set from this row lands on; null where none can be set. */
  target: LimitTarget | null
  /** The workspaces the row's sessions can be in: a key's one, a person's few. */
  workspaceIds?: string[]
  /** A person's row: the account their avatar is drawn from. */
  avatarUserId?: string | null
  /** The face they picked, if any. */
  avatarFace?: number | null
  /** A workspace's row, when its summary was read by model. */
  byModel?: readonly UsageGroupRow[]
  /** The row's spend could not be read: its figures are unknown, not zero. */
  isUnread?: boolean
  /**
   * The summary has no row of its own for this spender and folded its
   * smallest into Other, so their figures are unknown rather than zero.
   */
  isFolded?: boolean
  /** A row for someone on the roster, rather than spend nobody there owns. */
  isMember?: boolean
}

export type ReadState = "loading" | "failed" | "ready"

/** A read by what it holds, so one that is off or retrying is not settled. */
export const readState = (query: {
  data: unknown
  isError: boolean
}): ReadState =>
  query.data !== undefined ? "ready" : query.isError ? "failed" : "loading"

/** Usage rows that are one spender, by key: not the fold, not "nobody". */
const bySpender = (usage: readonly UsageGroupRow[]) =>
  new Map(
    usage
      .filter((row) => !row.is_other && row.key !== null)
      .map((row) => [row.key, row]),
  )

/** Whether the server folded some spenders into its `is_other` row. */
const hasFold = (usage: readonly UsageGroupRow[]) =>
  usage.some((row) => row.is_other)

const bySpend = (a: SpendRow, b: SpendRow) =>
  b.cost - a.cost || a.name.localeCompare(b.name)

/**
 * One row per active member, then one per spender the roster does not hold.
 * `isUnread`: the usage read has not answered, so no figure is known.
 * `isRosterRead`: whether a spender missing from the roster has left it.
 */
export function peopleRows(
  usage: readonly UsageGroupRow[],
  members: readonly OrganizationMember[],
  limitsFor: (member: OrganizationMember) => RowLimits,
  canSetLimits: boolean,
  { isUnread = false, isRosterRead = true } = {},
): SpendRow[] {
  const spent = bySpender(usage)
  const isFoldedRead = hasFold(usage)
  const claimed = new Set<string | null>()
  const rows: SpendRow[] = members
    .filter((member) => member.status === "active")
    .map((member) => {
      const key = member.attribution_user_id ?? member.user_id ?? null
      const row =
        (key ? spent.get(key) : undefined) ??
        (member.user_id ? spent.get(member.user_id) : undefined)
      if (row) claimed.add(row.key)
      const limits = limitsFor(member)
      return {
        id: `person:${member.organization_member_id ?? key}`,
        filterKey: key,
        name: memberName(member),
        cost: row?.cost ?? 0,
        requests: row?.requests ?? 0,
        tokens: row?.tokens ?? 0,
        limits,
        workspaceIds: (member.workspaces ?? []).map(
          (place) => place.workspace_id,
        ),
        avatarUserId: member.user_id ?? null,
        avatarFace: member.avatar_face,
        isUnread,
        isFolded: isFoldedRead && !row,
        isMember: true,
        target:
          canSetLimits && !limits.isUnknown && member.organization_member_id
            ? {
                scope: "org_member",
                scopeId: member.organization_member_id,
                name: memberName(member),
                own: limits.own,
              }
            : null,
      }
    })
  // Spend with nobody on the roster behind it: someone who left, a service
  // account, the Playground. Listed so the rows still add up to the total.
  for (const row of usage) {
    if (row.cost <= 0 || (!row.is_other && claimed.has(row.key))) continue
    const isNobody = row.is_other || row.key === null
    rows.push({
      id: row.is_other ? "user:other" : `user:${row.key ?? "none"}`,
      filterKey: row.is_other ? null : row.key,
      name: row.is_other
        ? OTHER
        : row.key === null
          ? "No person"
          : row.label || `${row.key.slice(0, 8)}…`,
      cost: row.cost,
      requests: row.requests,
      tokens: row.tokens,
      limits: isNobody || isRosterRead ? NO_LIMITS : UNKNOWN_LIMITS,
      target: null,
    })
  }
  return rows.sort(bySpend)
}

/**
 * One row per key the caller lists, then one per spending key it does not:
 * a teammate's, or one deleted since. `limitsFor` takes a key's id, so either
 * kind reads its own limit.
 */
export function keyRows(
  usage: readonly UsageGroupRow[],
  keys: readonly ApiKey[],
  ownerOf: (key: ApiKey) => string | undefined,
  limitsFor: (keyId: string) => RowLimits,
  canSetLimits: boolean,
  {
    isUnread = false,
    now = Date.now(),
  }: {
    isUnread?: boolean
    /** A key past its expiry is not offered a limit: it cannot be used. */
    now?: number
  } = {},
): SpendRow[] {
  const spent = bySpender(usage)
  const isFoldedRead = hasFold(usage)
  const claimed = new Set<string | null>()
  const rows: SpendRow[] = []
  for (const key of keys) {
    const row = spent.get(key.id)
    if (row) claimed.add(row.key)
    // A revoked key that spent nothing in the window has nothing to say.
    if (!key.is_active && !row) continue
    const limits = limitsFor(key.id)
    const name = keyName(key)
    rows.push({
      id: `key:${key.id}`,
      filterKey: key.id,
      name,
      detail: ownerOf(key),
      cost: row?.cost ?? 0,
      requests: row?.requests ?? 0,
      tokens: row?.tokens ?? 0,
      limits,
      workspaceIds: [key.workspace_id],
      isUnread,
      isFolded: isFoldedRead && !row,
      target:
        canSetLimits && keyState(key, now) === "active" && !limits.isUnknown
          ? { scope: "api_token", scopeId: key.id, name, own: limits.own }
          : null,
    })
  }
  for (const row of usage) {
    if (row.cost <= 0 || (!row.is_other && claimed.has(row.key))) continue
    rows.push({
      id: row.is_other ? "key:other" : `key:${row.key ?? "none"}`,
      filterKey: row.is_other ? null : row.key,
      name: row.is_other
        ? OTHER
        : row.key === null
          ? "No key"
          : row.label || "A deleted key",
      detail: !row.is_other && row.key === null ? "The Playground" : undefined,
      cost: row.cost,
      requests: row.requests,
      tokens: row.tokens,
      limits: row.is_other || row.key === null ? NO_LIMITS : limitsFor(row.key),
      target: null,
    })
  }
  return rows.sort(bySpend)
}

/**
 * One row per value of a breakdown with spend, its fold as "Other", sorted by
 * spend. `named` names a value, or the requests that recorded none (`null`).
 */
function breakdownRows(
  usage: readonly UsageGroupRow[],
  kind: string,
  named: (key: string | null) => { name: string; detail?: string },
): SpendRow[] {
  return usage
    .filter((row) => row.cost > 0 || row.requests > 0)
    .map((row) => ({
      // The fold and the requests that recorded none take ids no value can:
      // an instance an operator names "other" or "none" is still its own row.
      id: row.is_other
        ? `${kind}::other`
        : row.key === null
          ? `${kind}::none`
          : `${kind}:${row.key}`,
      filterKey: row.is_other ? null : row.key,
      ...(row.is_other ? { name: OTHER } : named(row.key)),
      cost: row.cost,
      requests: row.requests,
      tokens: row.tokens,
      limits: NO_LIMITS,
      target: null,
    }))
    .sort(bySpend)
}

/** One row per model, its ids on every provider summed (`groupModels`). */
export function modelRows(usage: readonly UsageGroupRow[]): SpendRow[] {
  return groupModels(usage)
    .filter((model) => model.cost > 0 || model.requests > 0)
    .map((model) => ({
      id: model.id,
      filterKey: model.spellings[0] ?? null,
      spellings: model.spellings,
      name: model.name,
      cost: model.cost,
      requests: model.requests,
      tokens: model.tokens,
      limits: NO_LIMITS,
      target: null,
    }))
    .sort(bySpend)
}

/**
 * One row per provider the window's requests went to, by the name a usage row
 * records: the provider, or the instance an operator named. Requests that
 * recorded none (imported without one, refused before a provider was picked,
 * or written before the column existed) are one row of their own, so the rows
 * still add up to the total.
 */
export function providerRows(usage: readonly UsageGroupRow[]): SpendRow[] {
  return breakdownRows(usage, "provider", (key) =>
    key === null
      ? { name: "Unknown provider", detail: "Not recorded" }
      : { name: providerDisplayName(key) },
  )
}

/** What a row's filter takes: a model's every id, else its one key. */
export function filterValues(row: SpendRow): string[] {
  return row.spellings ?? (row.filterKey === null ? [] : [row.filterKey])
}

export interface Spend {
  totals: UsageTotals | undefined
  /** The window before this one, or null where there is nothing to compare. */
  previousCost: number | undefined
  series: UsageSeriesPoint[]
  people: SpendRow[]
  keys: SpendRow[]
  models: SpendRow[]
  providers: SpendRow[]
  /** The organization's own limit, with its edit target for a manager. */
  organization: { limits: RowLimits; target: LimitTarget | null }
  /** Every ceiling a manager may read; empty for anyone else. */
  ceilings: OrganizationSpendCeiling[]
  /**
   * The reads the rows stand on. `limits` is the ceilings, and is ready with
   * none for anyone not shown limits.
   */
  reads: { usage: ReadState; roster: ReadState; limits: ReadState }
  isLoading: boolean
  error: unknown
}

export function useSpend(
  window: SpendWindow,
  {
    workspaceId,
    userId,
    isEnabled = true,
    resetEndedAt,
    withPrevious = true,
  }: {
    workspaceId?: string
    /** Narrows every figure to one person: their keys, their models. */
    userId?: string
    isEnabled?: boolean
    /** Reads the window before this one, for `previousCost`. */
    withPrevious?: boolean
    /**
     * Reads every ceiling as of this time, so one whose period has ended counts
     * as nothing spent.
     */
    resetEndedAt: number
  },
): Spend {
  const canSetLimits = useIsOrganizationManager()
  const context = useOrganizationContext().data
  const scope = {
    workspace_id: workspaceId,
    user_id: userId ? [userId] : undefined,
  }
  const current = useUsageSummary(
    { ...scope, ...usageRange(window) },
    window.bucket,
    ["user", "api_key", "model", "provider"],
    isEnabled,
    "organization",
  )
  const previous = useUsageSummary(
    {
      ...scope,
      start_date: window.previousStart,
      end_date: window.previousEnd,
    },
    "day",
    NO_BREAKDOWNS,
    isEnabled && withPrevious,
    "organization",
  )
  const members = useOrganizationMembers()
  const keys = useKeys()
  // Owners and admins only: anyone else is refused the read, so it is not made.
  const ceilings = useAllSpendCeilings(canSetLimits)
  const rawCeilings = canSetLimits ? (ceilings.data ?? []) : []
  const ceilingList = resetEndedPeriods(rawCeilings, resetEndedAt)
  const memberList = members.data ?? []
  const reads: Spend["reads"] = {
    usage: readState(current),
    roster: readState(members),
    limits: canSetLimits ? readState(ceilings) : "ready",
  }
  const isUnread = reads.usage !== "ready"
  // Unread ceilings leave every limit unknown, and nothing to set one from.
  const limitsOf = (scope: LimitScope, id: string | null | undefined) =>
    reads.limits === "ready" ? limitsOn(ceilingList, scope, id) : UNKNOWN_LIMITS

  const nameByUser = new Map<string, string>()
  for (const member of memberList) {
    for (const id of [member.user_id, member.attribution_user_id]) {
      if (id) nameByUser.set(id, memberName(member))
    }
  }

  const summary = current.data
  const organizationId = context?.organization?.id
  const organizationLimits = limitsOf("organization", organizationId)
  const priorCost = previous.data?.totals.cost ?? 0

  return {
    totals: summary?.totals,
    // A previous window the record barely covers makes every change read as
    // +2,000%. Below a quarter of this window's spend there is no comparison.
    previousCost:
      summary && priorCost >= summary.totals.cost * 0.25 && priorCost > 0
        ? priorCost
        : undefined,
    series: summary?.series ?? [],
    people: peopleRows(
      summary?.by_user ?? [],
      memberList,
      (member) =>
        reads.limits === "ready"
          ? limitsOnPerson(ceilingList, member)
          : UNKNOWN_LIMITS,
      canSetLimits,
      { isUnread, isRosterRead: reads.roster === "ready" },
    ),
    keys: keyRows(
      summary?.by_api_key ?? [],
      keys.data ?? [],
      (key) => (key.user_id ? nameByUser.get(key.user_id) : "No owner"),
      (id) => limitsOf("api_token", id),
      canSetLimits,
      { isUnread, now: resetEndedAt },
    ),
    models: modelRows(summary?.by_model ?? []),
    providers: providerRows(summary?.by_provider ?? []),
    organization: {
      limits: organizationLimits,
      target:
        canSetLimits && organizationId && !organizationLimits.isUnknown
          ? {
              scope: "organization",
              scopeId: organizationId,
              name: context?.organization?.name ?? "Organization",
              own: organizationLimits.own,
            }
          : null,
    },
    ceilings: ceilingList,
    reads,
    isLoading:
      current.isLoading ||
      previous.isLoading ||
      members.isLoading ||
      keys.isLoading ||
      (canSetLimits && ceilings.isLoading),
    error:
      current.error ??
      previous.error ??
      members.error ??
      keys.error ??
      (canSetLimits ? ceilings.error : null),
  }
}

/**
 * The organization's own limit as Home reads it: the ceiling over every
 * provider, with a period that has ended counted as nothing spent. Null until
 * the ceilings are read, and for anyone not shown limits.
 */
export function useOrganizationLimit(): AppliedLimit | null {
  const isManager = useIsOrganizationManager()
  const organizationId = useOrganizationContext().data?.organization?.id
  const ceilings = useAllSpendCeilings(isManager)
  if (!isManager || ceilings.data === undefined) return null
  // As of the read rather than a clock taken once: the rail that shows this
  // stays mounted across a reset, and every refetch moves the read forward.
  const current = resetEndedPeriods(ceilings.data, ceilings.dataUpdatedAt)
  return limitsOn(current, "organization", organizationId).binding
}

/**
 * Each workspace's spend over the window. The summary has no workspace
 * breakdown, so this asks once per workspace: a handful, read in parallel.
 */
export function useWorkspaceSpend(
  window: SpendWindow,
  isEnabled: boolean,
  /** As `useSpend`'s `resetEndedAt`: a ceiling whose period ended counts as nothing spent. */
  resetEndedAt: number,
): { rows: SpendRow[]; isLoading: boolean; error: unknown } {
  const canSetLimits = useIsOrganizationManager()
  const workspaces = useWorkspaces()
  const ceilings = useAllSpendCeilings(canSetLimits)
  const rawCeilings = canSetLimits ? (ceilings.data ?? []) : []
  const ceilingList = resetEndedPeriods(rawCeilings, resetEndedAt)
  const workspaceList = workspaces.data ?? []
  const isLimitsRead = !canSetLimits || readState(ceilings) === "ready"
  const summaries = useWorkspaceSpendSummaries(
    workspaceList.map((workspace) => workspace.id),
    window.start,
    isEnabled,
    window.end,
  )
  const rows = workspaceList
    .map((workspace, index): SpendRow => {
      const summary = summaries[index].data?.totals
      const limits = isLimitsRead
        ? limitsOn(ceilingList, "workspace", workspace.id)
        : UNKNOWN_LIMITS
      return {
        id: `workspace:${workspace.id}`,
        filterKey: workspace.id,
        name: workspace.name,
        detail: workspace.description || undefined,
        cost: summary?.cost ?? 0,
        requests: summary?.request_count ?? 0,
        // Billed tokens, cache reads included, as the other breakdowns count.
        tokens: billedTokenTotal(summary) ?? 0,
        limits,
        byModel: summaries[index].data?.by_model,
        isUnread: summary === undefined,
        target:
          canSetLimits && !limits.isUnknown
            ? {
                scope: "workspace",
                scopeId: workspace.id,
                name: workspace.name,
                own: limits.own,
              }
            : null,
      }
    })
    .sort(bySpend)
  return {
    rows,
    isLoading:
      workspaces.isLoading ||
      summaries.some((one) => one.isLoading) ||
      (canSetLimits && ceilings.isLoading),
    error:
      workspaces.error ??
      summaries.find((one) => one.error)?.error ??
      (canSetLimits ? ceilings.error : null),
  }
}
