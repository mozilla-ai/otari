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
    expect(toStoredRate(2, "request")).toBe(2000)
    // Rounded to the micro-dollar, not left as a float artifact.
    expect(toStoredRate(0.07, "image")).toBe(70)
    expect(toStoredRate(0.25, "token")).toBe(0.25)
  })

  it("shows a stored rate back in the unit it was entered in", () => {
    expect(toEnteredRate(2000, "request")).toBe(2)
    expect(toEnteredRate(0.25, "token")).toBe(0.25)
  })

  it("reads an unknown or absent unit as tokens", () => {
    expect(pricingUnitOf("request")).toBe("request")
    expect(pricingUnitOf("seconds")).toBe("token")
    expect(pricingUnitOf(undefined)).toBe("token")
  })

  it("reads a retired plural name as the current one", () => {
    expect(pricingUnitOf("tokens")).toBe("token")
    expect(pricingUnitOf("requests")).toBe("request")
    expect(pricingUnitOf("images")).toBe("image")
  })

  it("picks the unit an endpoint is billed in", () => {
    expect(pricingUnitForEndpoint("/v1/rerank")).toBe("request")
    expect(pricingUnitForEndpoint("/v1/images/generations")).toBe("image")
    expect(pricingUnitForEndpoint("/v1/images/edits")).toBe("image")
    expect(pricingUnitForEndpoint("/v1/audio/transcriptions")).toBe("request")
    expect(pricingUnitForEndpoint("/v1/audio/speech")).toBe("request")
    expect(pricingUnitForEndpoint("/v1/moderations")).toBe("request")
    expect(pricingUnitForEndpoint("/v1/chat/completions")).toBe("token")
    expect(pricingUnitForEndpoint(undefined)).toBe("token")
  })

  it("names the unit a rate is charged in", () => {
    expect(formatPricedRate(2000, "request")).toBe("$0.002 per request")
    expect(formatPricedRate(40000, "image")).toBe("$0.04 per image")
    expect(formatPricedRate(2.5, "token")).toBe("$2.50 / 1M")
  })
})
