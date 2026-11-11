import { describe, expect, it } from "vitest"

import {
  COLUMN_PRICING_UNIT_OPTIONS,
  formatOfferingPrice,
  formatPricedRate,
  hasOutputRate,
  PRICING_UNIT_OPTIONS,
  pricesAudio,
  pricingUnitForEndpoint,
  pricingUnitOf,
  ratesForUnit,
  toEnteredRate,
  toStoredRate,
  unitRateLabel,
} from "@/features/models/pricingUnit"

describe("pricingUnit", () => {
  it("stores a per-thousand request rate per million", () => {
    expect(toStoredRate(2, "request")).toBe(2000)
    // Rounded to the micro-dollar, not left as a float artifact.
    expect(toStoredRate(0.07, "image")).toBe(70)
    expect(toStoredRate(0.25, "token")).toBe(0.25)
    // A character rate is published per million, as a token rate is.
    expect(toStoredRate(15, "character")).toBe(15)
    // A megapixel, second or minute rate is entered per one, a search rate per thousand.
    expect(toStoredRate(0.03, "megapixel")).toBe(30000)
    expect(toStoredRate(0.06, "second")).toBe(60000)
    expect(toStoredRate(0.006, "minute")).toBe(6000)
    expect(toStoredRate(10, "web_search")).toBe(10000)
  })

  it("shows a stored rate back in the unit it was entered in", () => {
    expect(toEnteredRate(2000, "request")).toBe(2)
    expect(toEnteredRate(0.25, "token")).toBe(0.25)
    expect(toEnteredRate(15, "character")).toBe(15)
    expect(toEnteredRate(60000, "second")).toBe(0.06)
    expect(toEnteredRate(10000, "web_search")).toBe(10)
  })

  it("reads an unknown or absent unit as tokens", () => {
    expect(pricingUnitOf("request")).toBe("request")
    expect(pricingUnitOf("character")).toBe("character")
    expect(pricingUnitOf("megapixel")).toBe("megapixel")
    expect(pricingUnitOf("second")).toBe("second")
    expect(pricingUnitOf("minute")).toBe("minute")
    expect(pricingUnitOf("web_search")).toBe("web_search")
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
    expect(pricingUnitForEndpoint("/v1/audio/speech")).toBe("character")
    expect(pricingUnitForEndpoint("/v1/moderations")).toBe("request")
    expect(pricingUnitForEndpoint("/v1/videos")).toBe("second")
    expect(pricingUnitForEndpoint("/v1/chat/completions")).toBe("token")
    expect(pricingUnitForEndpoint(undefined)).toBe("token")
  })

  it("names the unit a rate is charged in", () => {
    expect(formatPricedRate(2000, "request")).toBe("$0.002 per request")
    expect(formatPricedRate(40000, "image")).toBe("$0.04 per image")
    expect(formatPricedRate(2.5, "token")).toBe("$2.50 / 1M")
    expect(formatPricedRate(15, "character")).toBe("$15.00 / 1M characters")
    expect(formatPricedRate(30000, "megapixel")).toBe("$0.03 per megapixel")
    expect(formatPricedRate(60000, "second")).toBe("$0.06 per second")
    expect(formatPricedRate(6000, "minute")).toBe("$0.006 per minute")
    expect(formatPricedRate(10000, "web_search")).toBe("$10.00 / 1K searches")
  })

  it("labels the one rate a character price carries", () => {
    expect(unitRateLabel("character")).toBe("Price per 1M characters")
    expect(unitRateLabel("image")).toBe("Price per 1,000 images")
    expect(unitRateLabel("request")).toBe("Price per 1,000 requests")
    expect(unitRateLabel("megapixel")).toBe("Price per megapixel")
    expect(unitRateLabel("second")).toBe("Price per second of video")
    expect(unitRateLabel("minute")).toBe("Audio in, per minute")
    expect(unitRateLabel("web_search")).toBe("Price per 1,000 searches")
  })

  it("offers every unit, and leaves a column-only form the ones the columns place", () => {
    expect(PRICING_UNIT_OPTIONS.map((option) => option.value)).toEqual([
      "token",
      "request",
      "image",
      "megapixel",
      "character",
      "minute",
      "second",
      "web_search",
    ])
    expect(
      COLUMN_PRICING_UNIT_OPTIONS.map((option) => option.value),
    ).not.toContain("second")
  })

  it("prices a second of video where a video is billed from", () => {
    // The per-million fields would store a per-second price on audio input.
    expect(ratesForUnit(0.06, "second")).toEqual({
      video: { output: { price: 0.06, unit: "second" } },
    })
    expect(ratesForUnit(2, "request")).toBeUndefined()
    expect(ratesForUnit(0.006, "minute")).toBeUndefined()
  })

  it("moves only the video price of a stored block, keeping its rules", () => {
    const multipliers = [{ factor: 2, when: { resolution: { equals: "4k" } } }]
    expect(
      ratesForUnit(0.08, "second", {
        video: { output: { price: 0.05, unit: "second", overrides: [] } },
        multipliers,
      }),
    ).toEqual({
      video: { output: { price: 0.08, unit: "second", overrides: [] } },
      multipliers,
    })
  })

  it("leaves a stored second of audio to the columns, which keep its slot", () => {
    expect(
      ratesForUnit(0.0001, "second", {
        audio: { input: { price: 0.0001, unit: "second" } },
      }),
    ).toBeUndefined()
  })

  it("moves the audio price of a block that prices a second of both, as the field shows it", () => {
    // The columns, and so the field, are read from audio before video.
    expect(
      ratesForUnit(0.0002, "second", {
        audio: { input: { price: 0.0001, unit: "second" } },
        video: { output: { price: 0.05, unit: "second" } },
      }),
    ).toBeUndefined()
  })

  it("moves the slot the field read a second from, keeping the video price", () => {
    const video = { output: { price: 0.05, unit: "second" } }
    expect(
      ratesForUnit(0.03, "second", {
        image: { output: { price: 0.02, unit: "second" } },
        video,
      }),
    ).toEqual({ image: { output: { price: 0.03, unit: "second" } }, video })
    expect(
      ratesForUnit(0.03, "second", {
        text: { input: { price: 0.02, unit: "second" } },
        video,
      }),
    ).toEqual({ text: { input: { price: 0.03, unit: "second" } }, video })
  })

  it("tells a block whose columns are read from audio", () => {
    expect(
      pricesAudio({ audio: { output: { price: 0.0003, unit: "second" } } }),
    ).toBe(true)
    expect(
      pricesAudio({
        image: { output: { price: 0.02, unit: "second" } },
        audio: { input: { price: 0.0001, unit: "second" } },
      }),
    ).toBe(false)
    expect(
      pricesAudio({ video: { output: { price: 0.05, unit: "second" } } }),
    ).toBe(false)
    expect(pricesAudio(undefined)).toBe(false)
  })

  it("gives a minute of audio an output rate, and no other unit but tokens", () => {
    expect(hasOutputRate("minute")).toBe(true)
    for (const unit of ["request", "image", "second", "character"] as const)
      expect(hasOutputRate(unit)).toBe(false)
  })

  it("names an offering's price in a line, in its unit", () => {
    expect(
      formatOfferingPrice({
        input_price_per_million: 0.5,
        output_price_per_million: 2,
        unit: "token",
      }),
    ).toBe("$0.50 in / $2.00 out")
    expect(
      formatOfferingPrice({
        input_price_per_million: 40000,
        output_price_per_million: 0,
        unit: "image",
      }),
    ).toBe("$0.04 per image")
    expect(formatOfferingPrice(null)).toBe("unpriced")
  })
})
