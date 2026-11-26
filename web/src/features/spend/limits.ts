/**
 * A spend limit as the person setting it thinks of it: an amount, a period,
 * and who it is on. otari stores that as a budget plus the ceiling that points
 * it at someone, so every write here makes or moves both and nobody on the
 * page has to know there are two.
 */

import { useRef, useState } from "react"
import type { OrganizationMember, OrganizationSpendCeiling } from "@/client"
import {
  useCreateOrganizationBudget,
  useCreateOrganizationSpendCeiling,
  useDeleteOrganizationBudget,
  useDeleteOrganizationSpendCeiling,
  useOrganizationBudgets,
  useUpdateOrganizationBudget,
  useUpdateOrganizationSpendCeiling,
} from "@/shared/api/budgets"
import { formatUsd } from "@/shared/helpers/format"

const DAY_S = 86_400

export type LimitScope =
  | "organization"
  | "workspace"
  | "org_member"
  | "api_token"

/**
 * How often a limit starts over. A day, week and month turn with the calendar;
 * an hour rolls from the first request, which is what stops an agent caught in
 * a loop without cutting off the rest of someone's day.
 */
export type LimitPeriod =
  | "hour"
  | "calendar_day"
  | "calendar_week"
  | "calendar_month"

export const LIMIT_PERIODS: { value: LimitPeriod; label: string }[] = [
  { value: "hour", label: "Hour" },
  { value: "calendar_day", label: "Day" },
  { value: "calendar_week", label: "Week" },
  { value: "calendar_month", label: "Month" },
]

export const HOUR_S = 3_600

const PERIOD_WORD: Record<LimitPeriod, string> = {
  hour: "hourly",
  calendar_day: "daily",
  calendar_week: "weekly",
  calendar_month: "monthly",
}

export interface AppliedLimit {
  ceiling: OrganizationSpendCeiling
  max: number
  /** The ceiling's own count, which is what the gateway enforces. */
  spent: number
  /** "a week", "a month", "every 10 days", or "" where it never resets. */
  per: string
  /** Where a limit that is not the row's own comes from: "in Platform". */
  origin?: string
}

export interface RowLimits {
  /** The limit closest to its cap: the one that bites first. */
  binding: AppliedLimit | null
  /** The row's own limit, which is the one its dialog changes. */
  own: AppliedLimit | null
  /** Every limit that reaches this row. */
  count: number
  /** The limits could not be read, so none is not the same as no limit. */
  isUnknown?: boolean
}

export const NO_LIMITS: RowLimits = { binding: null, own: null, count: 0 }

export const UNKNOWN_LIMITS: RowLimits = { ...NO_LIMITS, isUnknown: true }

/** How often a ceiling resets, as it reads after an amount. */
export function perWords(
  ceiling: Pick<
    OrganizationSpendCeiling,
    "reset_alignment" | "budget_duration_sec"
  >,
): string {
  if (ceiling.reset_alignment === "calendar_day") return "a day"
  if (ceiling.reset_alignment === "calendar_week") return "a week"
  if (ceiling.reset_alignment === "calendar_month") return "a month"
  const seconds = ceiling.budget_duration_sec
  if (!seconds) return ""
  if (seconds < DAY_S) {
    const hours = Math.max(1, Math.round(seconds / HOUR_S))
    return hours === 1 ? "an hour" : `every ${hours} hours`
  }
  const days = Math.round(seconds / DAY_S)
  if (days === 1) return "a day"
  if (days === 7) return "a week"
  if (days >= 28 && days <= 31) return "a month"
  return `every ${days} days`
}

function applied(
  ceiling: OrganizationSpendCeiling,
  origin?: string,
): AppliedLimit | null {
  // $0 is a limit: nothing may be spent. Only a null amount caps nothing.
  const max = ceiling.max_budget
  if (max == null || ceiling.provider_key_id !== null) return null
  return {
    ceiling,
    max,
    spent: ceiling.current_spend,
    per: perWords(ceiling),
    origin,
  }
}

/** How much of a limit is used; a $0 limit is used up before it starts. */
export const usedShare = (limit: { spent: number; max: number }): number =>
  limit.max > 0 ? limit.spent / limit.max : 1

const share = usedShare

function summarize(own: AppliedLimit | null, rest: AppliedLimit[]): RowLimits {
  const all = own ? [own, ...rest] : rest
  if (all.length === 0) return NO_LIMITS
  const binding = all.reduce((worst, one) =>
    share(one) > share(worst) ? one : worst,
  )
  return { binding, own, count: all.length }
}

/**
 * Ceilings as they stand at `now`. The gateway resets a ceiling on the next
 * request that reaches it, so one whose period ended with nothing since still
 * carries last period's spend: read at `now` it has spent nothing.
 */
export function resetEndedPeriods(
  ceilings: readonly OrganizationSpendCeiling[],
  now: number,
): OrganizationSpendCeiling[] {
  return ceilings.map((one) =>
    one.period_end && Date.parse(one.period_end) < now && one.current_spend > 0
      ? { ...one, current_spend: 0 }
      : one,
  )
}

/** A scope's own limit: the organization's, a workspace's or a key's. */
export function limitsOn(
  ceilings: readonly OrganizationSpendCeiling[],
  scope: LimitScope,
  scopeId: string | null | undefined,
): RowLimits {
  if (!scopeId) return NO_LIMITS
  const own = ceilings
    .filter((one) => one.scope_type === scope && one.scope_id === scopeId)
    .map((one) => applied(one))
    .find((one) => one !== null)
  return summarize(own ?? null, [])
}

/**
 * Every limit that reaches a person: their own, and the one each workspace
 * gives its members. The workspace ones are shared defaults, so they are shown
 * on the row and changed where the default is set.
 */
export function limitsOnPerson(
  ceilings: readonly OrganizationSpendCeiling[],
  member: OrganizationMember,
): RowLimits {
  const own = limitsOn(
    ceilings,
    "org_member",
    member.organization_member_id,
  ).own
  const places = new Map(
    (member.workspaces ?? []).map((place) => [
      place.workspace_member_id,
      place.workspace_name,
    ]),
  )
  const inherited = ceilings
    .filter(
      (one) =>
        one.scope_type === "workspace_member" && places.has(one.scope_id),
    )
    .map((one) => applied(one, `in ${places.get(one.scope_id)}`))
    .filter((one): one is AppliedLimit => one !== null)
  return summarize(own, inherited)
}

/** A share as Home prints it: whole numbers, and never a rounded-away zero. */
export function wholePct(fraction: number): string {
  if (fraction > 0 && fraction < 0.01) return "<1%"
  return `${Math.round(fraction * 100)}%`
}

/** "$25.62 of $30.00 a week", the whole of a limit in one line. */
export function limitLine(limit: AppliedLimit): string {
  const per = limit.per ? ` ${limit.per}` : ""
  return `${formatUsd(limit.spent)} of ${formatUsd(limit.max)}${per}`
}

export interface LimitTarget {
  scope: LimitScope
  scopeId: string
  /** Who the limit is on, for the budget's name in Rules: "Wei Zhang". */
  name: string
  own: AppliedLimit | null
}

/**
 * Sets, changes and removes a limit in one step each. A budget shared by
 * several ceilings is never edited in place: changing it would move everyone
 * else's limit too, so this one gets a budget of its own. An existing ceiling
 * is pointed at that budget rather than made again, so the scope is never
 * without a limit and what it has spent this period still counts.
 */
export function useLimitWriter(isEnabled = true) {
  const budgets = useOrganizationBudgets(isEnabled)
  const createBudget = useCreateOrganizationBudget()
  const updateBudget = useUpdateOrganizationBudget()
  const deleteBudget = useDeleteOrganizationBudget()
  const createCeiling = useCreateOrganizationSpendCeiling()
  const updateCeiling = useUpdateOrganizationSpendCeiling()
  const deleteCeiling = useDeleteOrganizationSpendCeiling()
  const [error, setError] = useState<unknown>()
  // Ceilings this writer has deleted. A retry must not delete one again, or
  // edit its budget: that limit is already gone.
  const deleted = useRef(new Set<string>())
  // The budget each ceiling this writer moved now names. The caller holds the
  // one it left until it reads the limits again, and editing that one in place
  // would change whoever still shares it.
  const moved = useRef(new Map<string, string>())

  const [isReading, setReading] = useState(false)
  // A list read on demand, for the rest of the write it was read for: the
  // render that holds `budgets.data` comes after the write has moved on.
  const readList = useRef<typeof budgets.data>(undefined)

  /**
   * Whether anything besides this limit names a budget, read first if the list
   * has not answered yet. A list that cannot be read, or that does not carry
   * the budget, counts as shared: the budget is kept, never edited in place.
   */
  const isShared = async (budgetId: string): Promise<boolean> => {
    let list = budgets.data ?? readList.current
    if (list === undefined) {
      setReading(true)
      try {
        list = (await budgets.refetch({ cancelRefetch: false })).data
        readList.current = list
      } finally {
        setReading(false)
      }
    }
    const budget = list?.find((one) => one.budget_id === budgetId)
    return budget === undefined || budget.ceiling_count > 1
  }

  const ownCeiling = (target: LimitTarget) => {
    const ceiling = target.own?.ceiling
    if (!ceiling || deleted.current.has(ceiling.id)) return undefined
    const budgetId = moved.current.get(ceiling.id)
    return budgetId ? { ...ceiling, budget_id: budgetId } : ceiling
  }

  /**
   * The limit on a new budget of its own. The new budget comes first, and goes
   * again if nothing ends up naming it; an old budget only this limit named
   * goes once the limit has left it.
   */
  const replace = async (
    target: LimitTarget,
    amount: number,
    period: LimitPeriod,
  ): Promise<boolean> => {
    setError(undefined)
    try {
      const existing = ownCeiling(target)
      const wasShared = existing ? await isShared(existing.budget_id) : true
      const budget = await createBudget.mutateAsync({
        name: `${target.name}, ${PERIOD_WORD[period]}`,
        max_budget: amount,
        ...(period === "hour"
          ? { budget_duration_sec: HOUR_S }
          : { reset_alignment: period }),
      })
      try {
        if (existing) {
          await updateCeiling.mutateAsync({
            id: existing.id,
            body: { budget_id: budget.budget_id },
          })
          moved.current.set(existing.id, budget.budget_id)
        } else {
          await createCeiling.mutateAsync({
            scope_type: target.scope,
            scope_id: target.scopeId,
            budget_id: budget.budget_id,
          })
        }
      } catch (caught) {
        await deleteBudget.mutateAsync(budget.budget_id).catch(() => undefined)
        throw caught
      }
      // Left behind if it cannot go: a budget nothing names caps nothing.
      if (existing && !wasShared) {
        await deleteBudget
          .mutateAsync(existing.budget_id)
          .catch(() => undefined)
      }
      return true
    } catch (caught) {
      setError(caught)
      return false
    }
  }

  /**
   * `period` is null for a limit whose period the form cannot name: its
   * amount changes in place, but one that needs a new budget needs a period.
   */
  const save = async (
    target: LimitTarget,
    amount: number,
    period: LimitPeriod | null,
  ): Promise<boolean> => {
    const existing = ownCeiling(target)
    if (!existing || (await isShared(existing.budget_id))) {
      if (period === null) {
        setError(new Error("Choose how often this limit resets."))
        return false
      }
      return replace(target, amount, period)
    }
    setError(undefined)
    try {
      await updateBudget.mutateAsync({
        id: existing.budget_id,
        body: { max_budget: amount },
      })
      return true
    } catch (caught) {
      setError(caught)
      return false
    }
  }

  const remove = async (target: LimitTarget): Promise<boolean> => {
    const existing = ownCeiling(target)
    if (!existing) return true
    setError(undefined)
    try {
      const wasShared = await isShared(existing.budget_id)
      await deleteCeiling.mutateAsync(existing.id)
      deleted.current.add(existing.id)
      if (!wasShared) await deleteBudget.mutateAsync(existing.budget_id)
      return true
    } catch (caught) {
      setError(caught)
      return false
    }
  }

  return {
    save,
    replace,
    remove,
    error,
    clearError: () => setError(undefined),
    isPending:
      isReading ||
      createBudget.isPending ||
      updateBudget.isPending ||
      deleteBudget.isPending ||
      createCeiling.isPending ||
      updateCeiling.isPending ||
      deleteCeiling.isPending,
  }
}
