import { describe, expect, it } from "vitest"
import type { HostedModel } from "@/client"

import {
  canSubmitDraft,
  createBody,
  draftFrom,
  EMPTY_DRAFT,
  parsePrice,
  repriceBody,
} from "./modelPriceDraft"

const draft = (overrides: Partial<typeof EMPTY_DRAFT>) => ({
  ...EMPTY_DRAFT,
  ...overrides,
})

describe("parsePrice", () => {
  it("reads blank as no rate rather than as zero", () => {
    expect(parsePrice("")).toEqual({ ok: true })
    expect(parsePrice("   ")).toEqual({ ok: true })
  })

  it("accepts a non-negative number, zero included", () => {
    expect(parsePrice("2.5")).toEqual({ ok: true, value: 2.5 })
    expect(parsePrice("0")).toEqual({ ok: true, value: 0 })
  })

  it("refuses what a pricing row cannot hold", () => {
    expect(parsePrice("-1").ok).toBe(false)
    expect(parsePrice("two dollars").ok).toBe(false)
    expect(parsePrice("Infinity").ok).toBe(false)
  })
})

describe("canSubmitDraft", () => {
  it("offers a model with no price at all", () => {
    expect(canSubmitDraft(draft({ model: "gpt-4o" }), false)).toBe(true)
  })

  it("needs a model name to offer", () => {
    expect(canSubmitDraft(draft({ model: "  " }), false)).toBe(false)
  })

  it("refuses half a pair: input and output land in one row", () => {
    expect(
      canSubmitDraft(draft({ model: "gpt-4o", input: "2.5" }), false),
    ).toBe(false)
    expect(
      canSubmitDraft(
        draft({ model: "gpt-4o", input: "2.5", output: "10" }),
        false,
      ),
    ).toBe(true)
  })

  it("refuses a cache rate with no base pair to ride on", () => {
    expect(
      canSubmitDraft(draft({ model: "gpt-4o", cacheRead: "0.25" }), false),
    ).toBe(false)
    expect(
      canSubmitDraft(
        draft({
          model: "gpt-4o",
          input: "2.5",
          output: "10",
          cacheRead: "0.25",
        }),
        false,
      ),
    ).toBe(true)
  })

  it("requires the pair on a reprice, which is nothing but a price", () => {
    expect(canSubmitDraft(draft({ model: "gpt-4o" }), true)).toBe(false)
    expect(
      canSubmitDraft(
        draft({ model: "gpt-4o", input: "3", output: "12" }),
        true,
      ),
    ).toBe(true)
  })

  it("needs only the input rate when the price is per request or image", () => {
    expect(
      canSubmitDraft(
        draft({ model: "rerank-v3", unit: "requests", input: "2" }),
        true,
      ),
    ).toBe(true)
    // The hidden token boxes are not read, whatever they still hold.
    expect(
      canSubmitDraft(
        draft({ model: "rerank-v3", unit: "images", input: "2", output: "x" }),
        false,
      ),
    ).toBe(true)
  })

  it("refuses anything that does not parse", () => {
    expect(
      canSubmitDraft(
        draft({ model: "gpt-4o", input: "2.5", output: "ten" }),
        false,
      ),
    ).toBe(false)
  })
})

describe("createBody", () => {
  it("omits blanks so an unpriced offering stays unpriced", () => {
    expect(createBody(draft({ model: "gpt-4o" }))).toEqual({ model: "gpt-4o" })
  })

  it("carries the pair and only the cache rates that were typed", () => {
    expect(
      createBody(
        draft({
          model: "gpt-4o",
          input: "2.5",
          output: "10",
          cacheWrite: "3.75",
        }),
      ),
    ).toEqual({
      model: "gpt-4o",
      input_price_per_million: 2.5,
      output_price_per_million: 10,
      cache_write_price_per_million: 3.75,
      unit: "tokens",
    })
  })

  it("sends a per-thousand request rate per million, with a zero output", () => {
    expect(
      createBody(
        draft({
          model: "rerank-v3",
          unit: "requests",
          input: "2",
          output: "10",
          cacheRead: "1",
        }),
      ),
    ).toEqual({
      model: "rerank-v3",
      input_price_per_million: 2000,
      output_price_per_million: 0,
      unit: "requests",
    })
  })
})

describe("repriceBody", () => {
  it("clears a blanked cache box explicitly, with null rather than omission", () => {
    expect(
      repriceBody(
        draft({ model: "gpt-4o", input: "3", output: "12", cacheRead: "0.3" }),
      ),
    ).toEqual({
      input_price_per_million: 3,
      output_price_per_million: 12,
      cache_read_price_per_million: 0.3,
      cache_write_price_per_million: null,
      unit: "tokens",
    })
  })

  it("clears the cache rates when the price moves to images", () => {
    expect(
      repriceBody(draft({ model: "gpt-image-1", unit: "images", input: "40" })),
    ).toEqual({
      input_price_per_million: 40000,
      output_price_per_million: 0,
      cache_read_price_per_million: null,
      cache_write_price_per_million: null,
      unit: "images",
    })
  })
})

describe("draftFrom", () => {
  it("seeds the boxes with the price in force, absent rates blank", () => {
    const row: HostedModel = {
      id: "m1",
      provider: "openai",
      model: "gpt-4o",
      input_price_per_million: 2.5,
      output_price_per_million: 10,
      cache_read_price_per_million: 0.25,
      cache_write_price_per_million: null,
      unit: "tokens",
      price_source: "defaults",
      enabled: true,
      created_at: "2026-09-16T00:00:00Z",
      updated_at: null,
    }
    expect(draftFrom(row)).toEqual({
      model: "gpt-4o",
      unit: "tokens",
      input: "2.5",
      output: "10",
      cacheRead: "0.25",
      cacheWrite: "",
    })
  })

  it("shows a per-request rate per thousand, the way it is entered", () => {
    const row: HostedModel = {
      id: "m2",
      provider: "cohere",
      model: "rerank-v3",
      input_price_per_million: 2000,
      output_price_per_million: 0,
      cache_read_price_per_million: null,
      cache_write_price_per_million: null,
      unit: "requests",
      price_source: "deployment",
      enabled: true,
      created_at: "2026-09-16T00:00:00Z",
      updated_at: null,
    }
    expect(draftFrom(row)).toMatchObject({
      unit: "requests",
      input: "2",
      output: "",
    })
  })

  it("leaves the hidden cache boxes blank for a price that is not per token", () => {
    const row: HostedModel = {
      id: "m3",
      provider: "openai",
      model: "dall-e-3",
      input_price_per_million: 40000,
      output_price_per_million: 0,
      cache_read_price_per_million: 0.25,
      cache_write_price_per_million: 3.75,
      unit: "images",
      price_source: "deployment",
      enabled: true,
      created_at: "2026-09-16T00:00:00Z",
      updated_at: null,
    }
    expect(draftFrom(row)).toMatchObject({ cacheRead: "", cacheWrite: "" })
  })
})
