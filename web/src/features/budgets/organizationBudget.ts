/**
 * The vocabulary the organization's Spend page is written in.
 *
 * Pure derivations, in their own module so the page and its dialogs share one
 * answer rather than three: what a budget's figure reads as, and how a ceiling's
 * scope is named on screen. The reset cycle has its own module, `resetCycle.ts`,
 * because it is one concept carried by six fields.
 */

import type { OrganizationSpendCeiling, Workspace } from "@/client"
import { formatNumber, formatUsd } from "@/shared/helpers/format"

/** The three caps a budget can hold, as every response shape carries them. */
type BudgetCaps = {
  max_budget: number | null
  token_limit: number | null
  request_limit: number | null
}

/** Whether a budget caps nothing on any axis, and so admits every request. */
export function hasNoLimit(budget: BudgetCaps): boolean {
  return (
    budget.max_budget == null &&
    budget.token_limit == null &&
    budget.request_limit == null
  )
}

/**
 * Every cap a budget holds, where holding none is uncapped rather than zero.
 *
 * All three axes, not the dollar one: a budget capping only tokens reads as
 * unlimited when the label is derived from `max_budget` alone, which is the
 * opposite of what it does. The surface a token-capped ceiling is inspected
 * from is the one place that must not say that.
 */
export function limitLabel(budget: BudgetCaps): string {
  // Not `formatUsd(0)`: a budget with no cap on an axis admits every request on
  // it, which is the opposite of what "$0.00" reads as.
  if (hasNoLimit(budget)) return "No limit"
  const caps: string[] = []
  if (budget.max_budget != null) caps.push(formatUsd(budget.max_budget))
  if (budget.token_limit != null) {
    caps.push(`${formatNumber(budget.token_limit)} tokens`)
  }
  if (budget.request_limit != null) {
    caps.push(`${formatNumber(budget.request_limit)} requests`)
  }
  return caps.join(" + ")
}

/**
 * What a ceiling caps, in words.
 *
 * `scope_id` is a bare id on the wire, so a name has to be resolved from
 * something the page already read. Workspaces and the organization itself have
 * one; a membership or an API key does not, and its kind plus the head of its id
 * is more honest than a name invented here. Those two kinds are not creatable on
 * this page (a member's ceiling is set on Members & roles), so what this renders
 * is either a row created here or one the otari-ai cutover wrote.
 */
export function scopeLabel(
  ceiling: Pick<OrganizationSpendCeiling, "scope_type" | "scope_id">,
  context: { organizationName: string; workspaces: readonly Workspace[] },
): string {
  switch (ceiling.scope_type) {
    case "organization":
      return `${context.organizationName} (whole organization)`
    case "workspace": {
      const workspace = context.workspaces.find(
        (candidate) => candidate.id === ceiling.scope_id,
      )
      return workspace ? `${workspace.name} (workspace)` : "A workspace"
    }
    case "workspace_member":
      return `A workspace member (${shortId(ceiling.scope_id)})`
    case "org_member":
      return `An organization member (${shortId(ceiling.scope_id)})`
    case "api_token":
      return `An API key (${shortId(ceiling.scope_id)})`
    default:
      return ceiling.scope_type
  }
}

function shortId(value: string): string {
  return value.length > 8 ? `${value.slice(0, 8)}…` : value
}

/** The spend counters a ceiling carries this period, as the list endpoint has them. */
type CeilingUsage = Pick<
  OrganizationSpendCeiling,
  | "current_spend"
  | "reserved_spend"
  | "current_tokens"
  | "reserved_tokens"
  | "current_requests"
  | "reserved_requests"
>

/**
 * The entity closest to its limit, and how much of the limit it has used.
 *
 * Each entity draws on its own allowance of the limit, so a budget has no
 * single pool to measure. The figure that means something is the tightest one:
 * at 1 an entity under this budget is being refused, which a sum or an average
 * across entities would hide behind the others' headroom.
 *
 * Every capped axis counts and the furthest along wins, because a request is
 * refused on whichever axis runs out first. Held spend counts as used, as it
 * does on the detail page's meters. `null` when there is nothing to measure: no
 * cap, or no entities.
 */
export function tightestUsage<T extends CeilingUsage>(
  budget: BudgetCaps,
  ceilings: readonly T[],
): { used: number; ceiling: T } | null {
  const share = (value: number, limit: number | null) =>
    limit == null ? 0 : limit > 0 ? value / limit : value > 0 ? Infinity : 0
  if (hasNoLimit(budget)) return null
  let tightest: { used: number; ceiling: T } | null = null
  for (const ceiling of ceilings) {
    const used = Math.max(
      share(ceiling.current_spend + ceiling.reserved_spend, budget.max_budget),
      share(
        ceiling.current_tokens + ceiling.reserved_tokens,
        budget.token_limit,
      ),
      share(
        ceiling.current_requests + ceiling.reserved_requests,
        budget.request_limit,
      ),
    )
    if (!tightest || used > tightest.used) tightest = { used, ceiling }
  }
  return tightest
}
