import type {
  Budget,
  OrganizationSpendCeiling,
  ProviderHealthResponse,
  UsageTotals,
} from "@/client"

// Attention-routing status for an overview tile / the system-status strip.
// "neutral" means "nothing to judge here" (no data, unlimited, none configured)
// and renders as a plain tile with no color, never as green/red.
export type Health = "ok" | "warn" | "alert" | "neutral"

// ---------- error rate ----------

// Thresholds (fractions): >=2% amber, >=10% red. See plan issue #302.
export const ERROR_WARN = 0.02
export const ERROR_ALERT = 0.1

export interface ErrorRateHealth {
  // null when there are no requests to derive a rate from (renders as "—").
  rate: number | null
  status: Health
}

export function errorRateHealth(
  totals: UsageTotals | undefined,
): ErrorRateHealth {
  if (!totals || totals.request_count === 0) {
    return { rate: null, status: "neutral" }
  }
  const rate = totals.error_count / totals.request_count
  const status: Health =
    rate >= ERROR_ALERT ? "alert" : rate >= ERROR_WARN ? "warn" : "ok"
  return { rate, status }
}

// ---------- provider health ----------

// any-unreachable => amber; nothing usable => red; none configured/known => neutral.
// A provider counted as `degraded` (reachable-but-no-model-discovery, issue #447)
// is a warning, not an outage, so it never on its own turns the tile red.
export function providerHealthStatus(
  health: ProviderHealthResponse | undefined,
): Health {
  if (!health || health.total === 0) return "neutral"
  if (health.healthy >= health.total) return "ok"
  if (health.healthy + health.degraded === 0) return "alert"
  return "warn"
}

// ---------- budget health ----------

// >=80% of allocation amber, >=100% red, for either signal below. What an
// allocation *is* differs between them, and each says so.
export const BUDGET_WARN = 0.8

export interface BudgetHealth {
  status: Health
  label: string
  overCount: number
  nearCount: number
  // The rows with a finite cap, which are the ones we can judge.
  cappedCount: number
  worst?: { name: string; spent: number; allocated: number; pct: number }
}

/** One judgeable row: what it is called, what it has spent, what it may spend. */
interface Allocation {
  name: string
  spent: number
  allocated: number
}

/** No row here has a utilization, so there is nothing to be healthy or not. */
function noneToJudge(label: string): BudgetHealth {
  return {
    status: "neutral",
    label,
    overCount: 0,
    nearCount: 0,
    cappedCount: 0,
  }
}

/**
 * The judgment, once rows are reduced to allocations. Shared by the two signals
 * below so the deployment strip and the tenant one cannot classify the same
 * utilization differently.
 */
function allocationHealth(capped: Allocation[]): BudgetHealth {
  let overCount = 0
  let nearCount = 0
  let worst: BudgetHealth["worst"]
  let worstPct = -1
  for (const row of capped) {
    // A zero allocation admits nothing, so anything spent against one is over
    // it. Reported as a full 100% rather than as the infinite ratio it really
    // is: the share has no finite value, and a cell reading "Infinity%" tells
    // the reader less than "Over budget" at 100% does. It is a floor, so a row
    // measurably further past its limit still wins `worst`.
    const pct =
      row.allocated > 0 ? row.spent / row.allocated : row.spent > 0 ? 1 : 0
    if (pct >= 1) overCount += 1
    else if (pct >= BUDGET_WARN) nearCount += 1
    if (pct > worstPct) {
      worstPct = pct
      worst = { ...row, pct }
    }
  }
  const status: Health = overCount > 0 ? "alert" : nearCount > 0 ? "warn" : "ok"
  const label =
    overCount > 0
      ? `${overCount} over limit`
      : nearCount > 0
        ? `${nearCount} near limit`
        : "All within budget"
  return {
    status,
    label,
    overCount,
    nearCount,
    cappedCount: capped.length,
    worst,
  }
}

/**
 * The deployment's own budgets, as the operator strip reads them.
 *
 * `max_budget` there is a PER-USER cap that users share, so the honest
 * allocation is cap * user_count (which is what BudgetsPage's UsageCell shows).
 * Unlimited caps and user-less budgets have no utilization to judge.
 */
export function budgetHealth(budgets: Budget[]): BudgetHealth {
  if (budgets.length === 0) {
    return noneToJudge("No budgets configured")
  }
  const capped = budgets
    .filter((b) => b.max_budget !== null && b.user_count > 0)
    .map((b) => ({
      name: b.name ?? b.budget_id,
      spent: b.total_spend,
      allocated: (b.max_budget as number) * b.user_count,
    }))
  if (capped.length === 0) {
    return noneToJudge("No capped budgets")
  }
  return allocationHealth(capped)
}

/**
 * The same judgment over the rows a tenant can read: their organization's spend
 * ceilings.
 *
 * A ceiling carries its own counters, so the allocation is `max_budget` itself
 * rather than a per-user cap times a roster, and what it is judged against is
 * `current_spend + reserved_spend`, the sum a ceiling actually refuses on.
 *
 * `manageable` is deliberately not consulted. It says whose figure this is, not
 * whose spend: a ceiling naming a budget the organization does not own is
 * enforcing against that organization today, so dropping it would let the page
 * read as uncapped.
 *
 * `nameOf` names a row for the reader, because a ceiling's own label is optional
 * and what it caps is an id on the wire.
 */
export function spendCeilingHealth(
  ceilings: readonly OrganizationSpendCeiling[],
  nameOf: (ceiling: OrganizationSpendCeiling) => string,
): BudgetHealth {
  if (ceilings.length === 0) {
    return noneToJudge("No spend ceilings configured")
  }
  const capped = ceilings
    .filter((c) => c.max_budget !== null)
    .map((c) => ({
      name: nameOf(c),
      spent: c.current_spend + c.reserved_spend,
      allocated: c.max_budget as number,
    }))
  if (capped.length === 0) {
    return noneToJudge("No ceiling caps spend")
  }
  return allocationHealth(capped)
}
