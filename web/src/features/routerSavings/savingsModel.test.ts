import { describe, expect, it } from "vitest"

import {
  basisOf,
  chartPoints,
  coverageLine,
  formatSaved,
  pricedCount,
  spentOn,
} from "@/features/routerSavings/savingsModel"
import { savingsAmounts, subagentSavings } from "@/tests/fixtures"

describe("basisOf", () => {
  it("names how a group's money was arrived at", () => {
    expect(basisOf(savingsAmounts({ measured: 2, estimated: 0 }))).toBe(
      "Measured",
    )
    expect(basisOf(savingsAmounts({ measured: 0, estimated: 3 }))).toBe(
      "Estimated",
    )
    expect(basisOf(savingsAmounts({ measured: 1, estimated: 1 }))).toBe("Mixed")
    expect(
      basisOf(savingsAmounts({ with_outcome: 2, measured: 0, estimated: 0 })),
    ).toBe("Not priced")
    // Nothing reported is not a report without a price.
    expect(basisOf(savingsAmounts({ decisions: 2, with_outcome: 0 }))).toBe(
      "Not reported",
    )
  })
})

describe("the money", () => {
  it("counts the recommendations among what was paid", () => {
    const amounts = savingsAmounts({ actual_cost: 2.5, router_cost: 0.1 })
    expect(spentOn(amounts)).toBeCloseTo(2.6)
    expect(pricedCount(savingsAmounts({ measured: 1, estimated: 3 }))).toBe(4)
  })

  it("keeps the sign of a loss", () => {
    expect(formatSaved(5.5)).toBe("$5.50")
    expect(formatSaved(-0.21)).toBe("-$0.21")
    expect(formatSaved(0)).toBe("$0.00")
  })
})

describe("coverageLine", () => {
  it("says which decisions the money counts and which it leaves out", () => {
    const { totals } = subagentSavings()
    expect(coverageLine(totals)).toBe(
      "1 measured, 3 estimated, 1 not reported, 1 not priced",
    )
  })

  it("leaves out a kind there is none of", () => {
    const { totals } = subagentSavings({
      totals: { measured: 0, no_outcome: 0, unpriced: 0 },
    })
    expect(coverageLine(totals)).toBe("3 estimated")
  })
})

describe("chartPoints", () => {
  it("stacks the saving on what was paid, and stacks nothing for a loss", () => {
    const points = chartPoints([
      {
        ...savingsAmounts({ actual_cost: 1, router_cost: 0.5, net_saved: 3 }),
        bucket_start: "2026-11-01T00:00:00Z",
      },
      {
        ...savingsAmounts({ actual_cost: 2, router_cost: 0, net_saved: -1 }),
        bucket_start: "2026-11-02T00:00:00Z",
      },
    ])
    expect(points).toEqual([
      { x: "2026-11-01T00:00:00Z", spent: 1.5, saved: 3 },
      { x: "2026-11-02T00:00:00Z", spent: 2, saved: 0 },
    ])
  })
})
