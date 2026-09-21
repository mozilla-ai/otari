import type {
  AllocationHealth,
  ProviderHealthResponse,
  UsageTotals,
  WorstAllocation,
} from "@/client"
import { shortBudgetId } from "@/features/budgets/budgetLabel"

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
 * The server's scan, turned into the strip this page draws.
 *
 * The gateway decides which row is worst and how many are over or near, because
 * that means reading every row (otari#1425). What stays here is the wording and
 * the percentage, which are presentation: putting them on the wire would make
 * the API own UI copy.
 *
 * `health` is null where the caller may not see the strip at all, which is not
 * the same as a strip with nothing in it, so both answer "neutral" but with
 * different words.
 */
export function allocationStrip(
  health: AllocationHealth | null | undefined,
  labels: {
    none: string
    noneCapped: string
    /**
     * What to call the worst row when nobody named it. A spend ceiling is named
     * after what it caps ("A workspace"), which the id fingerprint below cannot
     * say; a deployment budget has no scope and keeps the fingerprint.
     */
    nameOf?: (worst: WorstAllocation) => string
  },
): BudgetHealth {
  if (!health || health.total_count === 0) return noneToJudge(labels.none)
  if (health.capped_count === 0 || !health.worst) {
    return noneToJudge(labels.noneCapped)
  }
  const { worst } = health
  const status: Health =
    health.over_count > 0 ? "alert" : health.near_count > 0 ? "warn" : "ok"
  return {
    status,
    label:
      health.over_count > 0
        ? `${health.over_count} over limit`
        : health.near_count > 0
          ? `${health.near_count} near limit`
          : "All within budget",
    overCount: health.over_count,
    nearCount: health.near_count,
    cappedCount: health.capped_count,
    worst: {
      name:
        worst.name ??
        labels.nameOf?.(worst) ??
        // The id fingerprint the budgets list already shows, which is what the
        // labeler did for an unnamed budget.
        shortBudgetId(worst.budget_id),
      spent: worst.spent,
      allocated: worst.allocated,
      // A zero allowance admits nothing, so anything spent against one is over
      // it. Reported as a full 100% rather than the infinite ratio it really
      // is, which is also how the gateway ranks it.
      pct:
        worst.allocated > 0
          ? worst.spent / worst.allocated
          : worst.spent > 0
            ? 1
            : 0,
    },
  }
}
