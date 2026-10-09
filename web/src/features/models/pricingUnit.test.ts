import { describe, expect, it } from "vitest"

import {
  formatPricedRate,
  pricingUnitForEndpoint,
  pricingUnitOf,
  toEnteredRate,
  toStoredRate,
} from "@/features/models/pricingUnit"

describe("pricingUnit", () => {
  it("stores a per-thousand request rate per million", () => {
    expect(toStoredRate(2, "requests")).toBe(2000)
    // Rounded to the micro-dollar, not left as a float artifact.
    expect(toStoredRate(0.07, "images")).toBe(70)
    expect(toStoredRate(0.25, "tokens")).toBe(0.25)
  })

  it("shows a stored rate back in the unit it was entered in", () => {
    expect(toEnteredRate(2000, "requests")).toBe(2)
    expect(toEnteredRate(0.25, "tokens")).toBe(0.25)
  })

  it("reads an unknown or absent unit as tokens", () => {
    expect(pricingUnitOf("requests")).toBe("requests")
    expect(pricingUnitOf("seconds")).toBe("tokens")
    expect(pricingUnitOf(undefined)).toBe("tokens")
  })

  it("picks the unit an endpoint is billed in", () => {
    expect(pricingUnitForEndpoint("/v1/rerank")).toBe("requests")
    expect(pricingUnitForEndpoint("/v1/images/generations")).toBe("images")
    expect(pricingUnitForEndpoint("/v1/chat/completions")).toBe("tokens")
    expect(pricingUnitForEndpoint(undefined)).toBe("tokens")
  })

  it("names the unit a rate is charged in", () => {
    expect(formatPricedRate(2000, "requests")).toBe("$0.002 per request")
    expect(formatPricedRate(40000, "images")).toBe("$0.04 per image")
    expect(formatPricedRate(2.5, "tokens")).toBe("$2.50 / 1M")
  })
})
