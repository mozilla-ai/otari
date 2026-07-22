import { describe, expect, it } from "vitest";

import type { Budget, ProviderHealthResponse, UsageTotals } from "@/api/types";
import { budgetHealth, errorRateHealth, providerHealthStatus, toStatStatus } from "@/lib/overview";

function budget(over: Partial<Budget>): Budget {
  return {
    budget_id: "b",
    name: null,
    max_budget: 100,
    budget_duration_sec: null,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    user_count: 1,
    total_spend: 0,
    total_reserved: 0,
    ...over,
  };
}

function totals(over: Partial<UsageTotals>): UsageTotals {
  return {
    cost: 0,
    prompt_tokens: 0,
    completion_tokens: 0,
    total_tokens: 0,
    cache_read_tokens: 0,
    cache_write_tokens: 0,
    request_count: 0,
    error_count: 0,
    avg_latency_ms: null,
    ...over,
  };
}

describe("errorRateHealth", () => {
  it("is neutral with no requests (no divide-by-zero)", () => {
    expect(errorRateHealth(totals({ request_count: 0, error_count: 0 }))).toEqual({ rate: null, status: "neutral" });
    expect(errorRateHealth(undefined)).toEqual({ rate: null, status: "neutral" });
  });

  it("crosses amber at 2% and red at 10%", () => {
    expect(errorRateHealth(totals({ request_count: 1000, error_count: 5 })).status).toBe("ok"); // 0.5%
    expect(errorRateHealth(totals({ request_count: 1000, error_count: 20 })).status).toBe("warn"); // 2%
    expect(errorRateHealth(totals({ request_count: 1000, error_count: 100 })).status).toBe("alert"); // 10%
  });
});

describe("providerHealthStatus", () => {
  const h = (healthy: number, total: number): ProviderHealthResponse => ({
    providers: [],
    healthy,
    total,
    checked_at: null,
  });
  it("is neutral when none are known", () => {
    expect(providerHealthStatus(undefined)).toBe("neutral");
    expect(providerHealthStatus(h(0, 0))).toBe("neutral");
  });
  it("grades healthy/degraded/down", () => {
    expect(providerHealthStatus(h(3, 3))).toBe("ok");
    expect(providerHealthStatus(h(2, 3))).toBe("warn");
    expect(providerHealthStatus(h(0, 3))).toBe("alert");
  });
});

describe("budgetHealth", () => {
  it("is neutral with no budgets configured", () => {
    expect(budgetHealth([]).status).toBe("neutral");
    expect(budgetHealth([]).label).toBe("No budgets configured");
  });

  it("excludes unlimited caps and user-less budgets", () => {
    const result = budgetHealth([
      budget({ max_budget: null, total_spend: 9999 }),
      budget({ user_count: 0, total_spend: 9999 }),
    ]);
    expect(result.status).toBe("neutral");
    expect(result.cappedCount).toBe(0);
  });

  it("uses cap * user_count for allocation (per-user cap)", () => {
    // cap 10 * 2 users = 20 allocated; spend 25 => over.
    const result = budgetHealth([budget({ max_budget: 10, user_count: 2, total_spend: 25, name: "team" })]);
    expect(result.status).toBe("alert");
    expect(result.overCount).toBe(1);
    expect(result.worst).toEqual({ name: "team", spent: 25, allocated: 20, pct: 1.25 });
  });

  it("flags near-limit at 80% and picks the worst-off budget", () => {
    const result = budgetHealth([
      budget({ budget_id: "a", max_budget: 100, user_count: 1, total_spend: 50 }), // 50%
      budget({ budget_id: "b", max_budget: 100, user_count: 1, total_spend: 85 }), // 85% near
    ]);
    expect(result.status).toBe("warn");
    expect(result.nearCount).toBe(1);
    expect(result.worst?.name).toBe("b");
  });
});

describe("toStatStatus", () => {
  it("maps neutral to undefined and passes the rest through", () => {
    expect(toStatStatus("neutral")).toBeUndefined();
    expect(toStatStatus("ok")).toBe("ok");
    expect(toStatStatus("warn")).toBe("warn");
    expect(toStatStatus("alert")).toBe("alert");
  });
});
