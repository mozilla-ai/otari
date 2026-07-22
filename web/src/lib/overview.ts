import type { Budget, ProviderHealthResponse, UsageTotals } from "@/api/types";

// Attention-routing status for an overview tile / the system-status strip.
// "neutral" means "nothing to judge here" (no data, unlimited, none configured)
// and renders as a plain tile with no color, never as green/red.
export type Health = "ok" | "warn" | "alert" | "neutral";

// StatCard only carries ok/warn/alert (a colored accent); neutral maps to an
// un-statused (plain) tile.
export function toStatStatus(health: Health): "ok" | "warn" | "alert" | undefined {
  return health === "neutral" ? undefined : health;
}

// ---------- error rate ----------

// Thresholds (fractions): >=2% amber, >=10% red. See plan issue #302.
export const ERROR_WARN = 0.02;
export const ERROR_ALERT = 0.1;

export interface ErrorRateHealth {
  // null when there are no requests to derive a rate from (renders as "—").
  rate: number | null;
  status: Health;
}

export function errorRateHealth(totals: UsageTotals | undefined): ErrorRateHealth {
  if (!totals || totals.request_count === 0) {
    return { rate: null, status: "neutral" };
  }
  const rate = totals.error_count / totals.request_count;
  const status: Health = rate >= ERROR_ALERT ? "alert" : rate >= ERROR_WARN ? "warn" : "ok";
  return { rate, status };
}

// ---------- provider health ----------

// any-unreachable => amber; all-unreachable => red; none configured/known => neutral.
export function providerHealthStatus(health: ProviderHealthResponse | undefined): Health {
  if (!health || health.total === 0) return "neutral";
  if (health.healthy >= health.total) return "ok";
  if (health.healthy === 0) return "alert";
  return "warn";
}

// ---------- budget health ----------

// >=80% of allocation amber, >=100% red. `max_budget` is a PER-USER cap and users
// share a budget, so the honest allocation is cap * user_count (matches
// BudgetsPage's UsageCell). Unlimited caps and user-less budgets have no
// meaningful utilization and are excluded.
export const BUDGET_WARN = 0.8;

export interface BudgetHealth {
  status: Health;
  label: string;
  overCount: number;
  nearCount: number;
  // Budgets with a finite cap and at least one user (the ones we can judge).
  cappedCount: number;
  worst?: { name: string; spent: number; allocated: number; pct: number };
}

export function budgetHealth(budgets: Budget[]): BudgetHealth {
  if (budgets.length === 0) {
    return { status: "neutral", label: "No budgets configured", overCount: 0, nearCount: 0, cappedCount: 0 };
  }
  const capped = budgets.filter((b) => b.max_budget !== null && b.user_count > 0);
  if (capped.length === 0) {
    return { status: "neutral", label: "No capped budgets", overCount: 0, nearCount: 0, cappedCount: 0 };
  }

  let overCount = 0;
  let nearCount = 0;
  let worst: BudgetHealth["worst"];
  let worstPct = -1;
  for (const b of capped) {
    const allocated = (b.max_budget as number) * b.user_count;
    const pct = allocated > 0 ? b.total_spend / allocated : 0;
    if (pct >= 1) overCount += 1;
    else if (pct >= BUDGET_WARN) nearCount += 1;
    if (pct > worstPct) {
      worstPct = pct;
      worst = { name: b.name ?? b.budget_id, spent: b.total_spend, allocated, pct };
    }
  }

  const status: Health = overCount > 0 ? "alert" : nearCount > 0 ? "warn" : "ok";
  const label =
    overCount > 0
      ? `${overCount} over limit`
      : nearCount > 0
        ? `${nearCount} near limit`
        : "All within budget";
  return { status, label, overCount, nearCount, cappedCount: capped.length, worst };
}
