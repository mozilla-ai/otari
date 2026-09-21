import { describe, expect, it } from "vitest"

import type { AllocationHealth, ProviderHealthResponse } from "@/client"
import {
  allocationStrip,
  errorRateHealth,
  providerHealthStatus,
} from "@/features/overview/overview"
import { usageTotals } from "@/tests/fixtures"

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

const LABELS = {
  none: "No budgets configured",
  noneCapped: "No capped budgets",
}

function health(over: Partial<AllocationHealth> = {}): AllocationHealth {
  return {
    over_count: 0,
    near_count: 0,
    capped_count: 1,
    total_count: 1,
    worst: {
      budget_id: "11111111-2222-3333-4444-555555555555",
      name: "Monthly",
      spent: 50,
      allocated: 100,
      scope_type: null,
      scope_id: null,
    },
    ...over,
  }
}

describe("allocationStrip", () => {
  it("is neutral with nothing configured", () => {
    const result = allocationStrip(health({ total_count: 0 }), LABELS)

    expect(result.status).toBe("neutral")
    expect(result.label).toBe("No budgets configured")
  })

  it("tells nothing configured from nothing capped", () => {
    const result = allocationStrip(
      health({ capped_count: 0, worst: null }),
      LABELS,
    )

    expect(result.status).toBe("neutral")
    expect(result.label).toBe("No capped budgets")
  })

  it("is neutral where the caller may not see the strip at all", () => {
    // Withheld rather than empty, so the page says the same thing it says for
    // a deployment with no budgets rather than showing a false zero.
    expect(allocationStrip(null, LABELS).status).toBe("neutral")
    expect(allocationStrip(undefined, LABELS).label).toBe(
      "No budgets configured",
    )
  })

  it("derives the share from the worst row the server picked", () => {
    const result = allocationStrip(
      health({ worst: { ...health().worst!, spent: 75, allocated: 300 } }),
      LABELS,
    )

    expect(result.worst?.pct).toBe(0.25)
  })

  it("reads spend against an allowance of zero as a full share", () => {
    // It admits nothing, so anything spent is past it, and the share has no
    // finite value to render.
    const result = allocationStrip(
      health({
        over_count: 1,
        worst: { ...health().worst!, spent: 5, allocated: 0 },
      }),
      LABELS,
    )

    expect(result.worst?.pct).toBe(1)
    expect(result.status).toBe("alert")
  })

  it("names an unnamed row by its id fingerprint", () => {
    const result = allocationStrip(
      health({ worst: { ...health().worst!, name: null } }),
      LABELS,
    )

    expect(result.worst?.name).toBe("11111111")
  })

  it("names an unnamed row by what it caps, where it caps something", () => {
    // A spend ceiling nobody named is named after its scope. Falling through to
    // the id fingerprint here would put hex in the meter's accessible name.
    const result = allocationStrip(
      health({
        worst: {
          ...health().worst!,
          name: null,
          scope_type: "workspace",
          scope_id: "ws-1",
        },
      }),
      { ...LABELS, nameOf: () => "A workspace" },
    )

    expect(result.worst?.name).toBe("A workspace")
  })

  it("words the strip from the counts the server returned", () => {
    expect(allocationStrip(health({ over_count: 2 }), LABELS).label).toBe(
      "2 over limit",
    )
    expect(allocationStrip(health({ near_count: 1 }), LABELS).label).toBe(
      "1 near limit",
    )
    expect(allocationStrip(health(), LABELS).label).toBe("All within budget")
  })

  it("puts over-limit ahead of near-limit in the status", () => {
    const result = allocationStrip(
      health({ over_count: 1, near_count: 3 }),
      LABELS,
    )

    expect(result.status).toBe("alert")
  })
})
