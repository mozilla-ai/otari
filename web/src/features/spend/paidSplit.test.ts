import { describe, expect, it } from "vitest"
import { formatUsd } from "@/shared/helpers/format"
import { paidSplitLine } from "./paidSplit"

describe("paidSplitLine", () => {
  it("is empty where hosted providers served nothing", () => {
    expect(paidSplitLine(undefined)).toBe("")
    expect(paidSplitLine({ cost: 12, deployment_paid_cost: 0 })).toBe("")
  })

  it("splits the hosted cost from the own-key cost", () => {
    expect(paidSplitLine({ cost: 286.5, deployment_paid_cost: 22.42 })).toBe(
      `${formatUsd(22.42)} hosted · ${formatUsd(264.08)} own keys`,
    )
  })
})
