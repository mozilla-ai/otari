import { describe, expect, it } from "vitest"

import type { Budget, ProviderHealthResponse } from "@/client"
import {
  budgetHealth,
  errorRateHealth,
  providerHealthStatus,
  spendCeilingHealth,
} from "@/features/overview/overview"
import { organizationSpendCeiling, usageTotals } from "@/tests/fixtures"

function budget(over: Partial<Budget>): Budget {
  return {
    budget_id: "b",
    organization_id: null,
    name: null,
    max_budget: 100,
    token_limit: null,
    request_limit: null,
    reset_alignment: null,
    budget_duration_sec: null,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    user_count: 1,
    total_spend: 0,
    total_reserved: 0,
    ...over,
  }
}

const totals = usageTotals

describe("errorRateHealth", () => {
  it("is neutral with no requests (no divide-by-zero)", () => {
    expect(
      errorRateHealth(totals({ request_count: 0, error_count: 0 })),
    ).toEqual({ rate: null, status: "neutral" })
    expect(errorRateHealth(undefined)).toEqual({
      rate: null,
      status: "neutral",
    })
  })

  it("crosses amber at 2% and red at 10%", () => {
    expect(
      errorRateHealth(totals({ request_count: 1000, error_count: 5 })).status,
    ).toBe("ok") // 0.5%
    expect(
      errorRateHealth(totals({ request_count: 1000, error_count: 20 })).status,
    ).toBe("warn") // 2%
    expect(
      errorRateHealth(totals({ request_count: 1000, error_count: 100 })).status,
    ).toBe("alert") // 10%
  })
})

describe("providerHealthStatus", () => {
  const h = (
    healthy: number,
    total: number,
    degraded = 0,
  ): ProviderHealthResponse => ({
    providers: [],
    healthy,
    degraded,
    total,
    checked_at: null,
  })
  it("is neutral when none are known", () => {
    expect(providerHealthStatus(undefined)).toBe("neutral")
    expect(providerHealthStatus(h(0, 0))).toBe("neutral")
  })
  it("grades healthy/degraded/down", () => {
    expect(providerHealthStatus(h(3, 3))).toBe("ok")
    expect(providerHealthStatus(h(2, 3))).toBe("warn")
    expect(providerHealthStatus(h(0, 3))).toBe("alert")
  })
  it("treats a missing model listing as a warning, not an outage", () => {
    // otari#447: those providers can still serve requests, so red is wrong.
    expect(providerHealthStatus(h(0, 3, 3))).toBe("warn")
    expect(providerHealthStatus(h(1, 3, 1))).toBe("warn")
    // A genuine outage alongside a discovery gap is still an outage.
    expect(providerHealthStatus(h(0, 3, 0))).toBe("alert")
  })
})

describe("budgetHealth", () => {
  it("is neutral with no budgets configured", () => {
    expect(budgetHealth([]).status).toBe("neutral")
    expect(budgetHealth([]).label).toBe("No budgets configured")
  })

  it("excludes unlimited caps and user-less budgets", () => {
    const result = budgetHealth([
      budget({ max_budget: null, total_spend: 9999 }),
      budget({ user_count: 0, total_spend: 9999 }),
    ])
    expect(result.status).toBe("neutral")
    expect(result.cappedCount).toBe(0)
  })

  it("uses cap * user_count for allocation (per-user cap)", () => {
    // cap 10 * 2 users = 20 allocated; spend 25 => over.
    const result = budgetHealth([
      budget({ max_budget: 10, user_count: 2, total_spend: 25, name: "team" }),
    ])
    expect(result.status).toBe("alert")
    expect(result.overCount).toBe(1)
    expect(result.worst).toEqual({
      name: "team",
      spent: 25,
      allocated: 20,
      pct: 1.25,
    })
  })

  it("reads spend against a cap of zero as over, not as within budget", () => {
    // `max_budget` is `ge=0` on the wire, so a budget that admits nothing is a
    // real figure, and spend recorded before it was lowered to zero is a real
    // state. Dividing was the trap: it left the row at 0% and the strip
    // reporting "All within budget" over a cap that refuses every request.
    const result = budgetHealth([
      budget({ max_budget: 0, user_count: 2, total_spend: 5, name: "frozen" }),
    ])
    expect(result.status).toBe("alert")
    expect(result.overCount).toBe(1)
    expect(result.worst?.pct).toBe(1)

    // An untouched zero cap has nothing to report and stays on track.
    expect(
      budgetHealth([budget({ max_budget: 0, user_count: 2, total_spend: 0 })])
        .status,
    ).toBe("ok")
  })

  it("flags near-limit at 80% and picks the worst-off budget", () => {
    const result = budgetHealth([
      budget({
        budget_id: "a",
        max_budget: 100,
        user_count: 1,
        total_spend: 50,
      }), // 50%
      budget({
        budget_id: "b",
        max_budget: 100,
        user_count: 1,
        total_spend: 85,
      }), // 85% near
    ])
    expect(result.status).toBe("warn")
    expect(result.nearCount).toBe(1)
    expect(result.worst?.name).toBe("b")
  })
})

describe("spendCeilingHealth", () => {
  const named = (ceiling: { name: string | null }) => ceiling.name ?? "a scope"

  it("is neutral with nothing capped", () => {
    expect(spendCeilingHealth([], named).status).toBe("neutral")
    expect(
      spendCeilingHealth(
        [organizationSpendCeiling({ max_budget: null, current_spend: 9999 })],
        named,
      ).cappedCount,
    ).toBe(0)
  })

  it("judges spend plus what is reserved against the ceiling's own figure", () => {
    // A ceiling refuses on the sum, so the cell has to judge the sum. Its
    // `max_budget` is the pooled figure, not a per-user cap, so no roster
    // multiplies it the way `budgetHealth` multiplies a budget's.
    const result = spendCeilingHealth(
      [
        organizationSpendCeiling({
          name: "Staging cap",
          max_budget: 250,
          current_spend: 180,
          reserved_spend: 20,
        }),
      ],
      named,
    )
    expect(result.status).toBe("warn")
    expect(result.worst).toEqual({
      name: "Staging cap",
      spent: 200,
      allocated: 250,
      pct: 0.8,
    })
  })

  it("counts a ceiling the organization may not edit", () => {
    // `manageable` is descriptive, never a permission: the row is enforcing
    // against this organization whoever set its figure, so it is judged.
    const result = spendCeilingHealth(
      [
        organizationSpendCeiling({
          name: "Deployment cap",
          manageable: false,
          max_budget: 100,
          current_spend: 150,
        }),
      ],
      named,
    )
    expect(result.status).toBe("alert")
    expect(result.overCount).toBe(1)
    expect(result.worst?.name).toBe("Deployment cap")
  })

  it("reads spend against a ceiling of zero as over", () => {
    const result = spendCeilingHealth(
      [
        organizationSpendCeiling({
          name: "Frozen",
          max_budget: 0,
          current_spend: 5,
        }),
      ],
      named,
    )
    expect(result.status).toBe("alert")
    expect(result.overCount).toBe(1)
    // 100%, not Infinity: the share has no finite value, and the severity word
    // beside it is what says the cap was exceeded rather than reached.
    expect(result.worst?.pct).toBe(1)
  })

  it("picks the worst-off ceiling", () => {
    const result = spendCeilingHealth(
      [
        organizationSpendCeiling({
          id: "a",
          max_budget: 100,
          current_spend: 10,
        }),
        organizationSpendCeiling({
          id: "b",
          name: "tightest",
          max_budget: 100,
          current_spend: 90,
        }),
      ],
      named,
    )
    expect(result.worst?.name).toBe("tightest")
    expect(result.nearCount).toBe(1)
  })
})
