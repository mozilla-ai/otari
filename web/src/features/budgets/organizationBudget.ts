/**
 * The vocabulary the organization's Spend page is written in.
 *
 * Pure derivations, in their own module so the two cards and their dialogs share
 * one answer rather than three: what a period is called, what a budget's figure
 * reads as, and how a ceiling's scope is named on screen.
 */

import type {
  CreateOrganizationBudget,
  OrganizationBudget,
  OrganizationSpendCeiling,
  Workspace,
} from "@/client"
import { formatNumber, formatUsd } from "@/shared/helpers/format"

/**
 * The calendar boundaries the API accepts, derived from the generated client
 * rather than restated here.
 *
 * Restating them as `string` is what broke the build once: the endpoint narrowed
 * `reset_alignment` to a three-value enum and every local `string` stopped
 * assigning to it. Deriving the type means the next change to that enum is a
 * type error at the two places that construct one, not a widening that compiles
 * and sends a value the server refuses.
 */
export type ResetAlignment = NonNullable<
  CreateOrganizationBudget["reset_alignment"]
>

/**
 * The periods this page offers, as calendar boundaries rather than seconds.
 *
 * The API accepts either (`reset_alignment` or `budget_duration_sec`, never
 * both), and this surface deliberately only writes the first. A duration is a
 * rolling integer measured from the last reset, so "86400" does not mean "resets
 * at midnight", it means "at least 24 hours, restarted on the next request":
 * on a quiet workspace the reset walks through the morning and stays there. And
 * a month has no duration at all, since 30 days is a 1.5 percent more generous
 * product than a calendar month. An admin picking "Monthly" means the calendar,
 * so that is what is stored.
 *
 * A ceiling created by the deployment surface may still carry a duration, which
 * is why `periodLabel` reads both.
 */
export const PERIOD_OPTIONS: readonly {
  value: string
  label: string
  alignment: ResetAlignment | undefined
}[] = [
  { value: "none", label: "No reset", alignment: undefined },
  { value: "calendar_day", label: "Daily", alignment: "calendar_day" },
  { value: "calendar_week", label: "Weekly", alignment: "calendar_week" },
  { value: "calendar_month", label: "Monthly", alignment: "calendar_month" },
]

const ALIGNMENT_LABELS: Record<string, string> = {
  calendar_day: "Daily, at UTC midnight",
  calendar_week: "Weekly, Monday 00:00 UTC",
  calendar_month: "Monthly, the 1st at 00:00 UTC",
}

const HOUR = 3_600
const DAY = 86_400

/** Which `PERIOD_OPTIONS` value a stored budget corresponds to, for the form. */
export function periodValue(budget: OrganizationBudget | undefined): string {
  if (!budget) return "calendar_month"
  if (budget.reset_alignment) return budget.reset_alignment
  // A duration-based budget has no option of its own here, so the form opens on
  // "No reset" rather than silently proposing to convert it. Saving would then
  // clear the duration, which the field description says.
  return "none"
}

/** How a period reads in a table cell, for either way a budget can carry one. */
export function periodLabel(
  budget: Pick<OrganizationBudget, "reset_alignment" | "budget_duration_sec">,
): string {
  if (budget.reset_alignment) {
    return ALIGNMENT_LABELS[budget.reset_alignment] ?? budget.reset_alignment
  }
  const seconds = budget.budget_duration_sec
  if (seconds === null || seconds === undefined) return "Never"
  if (seconds % DAY === 0) {
    const days = seconds / DAY
    return `Every ${days} ${days === 1 ? "day" : "days"}, from the last reset`
  }
  if (seconds % HOUR === 0) {
    const hours = seconds / HOUR
    return `Every ${hours} ${hours === 1 ? "hour" : "hours"}, from the last reset`
  }
  return `Every ${seconds}s, from the last reset`
}

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

/** How a budget is named on screen: its label, else the head of its id. */
export function budgetLabel(
  budget: Pick<OrganizationBudget, "name" | "budget_id">,
): string {
  return budget.name ?? budget.budget_id.split("-")[0]
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
