import { describe, expect, it } from "vitest"

import { pricingResponse } from "@/tests/fixtures"

import {
  GATEWAY_ROWS,
  hasUnpricedRow,
  isSetInConfig,
  listInstanceRows,
  pairWithRates,
  parsePerCallPrice,
  type RateRead,
  toPerCall,
} from "./toolRates"

describe("listInstanceRows", () => {
  it("lists the configured instances a stored one does not shadow, then the stored", () => {
    const rows = listInstanceRows("search", {
      stored: [
        {
          name: "exa",
          kind: "search",
          provider: "exa",
          options: {},
          decryptable: true,
          shadows_config: true,
        },
      ],
      config: [
        {
          name: "exa",
          kind: "search",
          provider: "exa",
          has_api_key: true,
          shadowed: true,
        },
        {
          name: "brave",
          kind: "search",
          provider: "brave",
          has_api_key: true,
          shadowed: false,
        },
      ],
    })

    expect(
      rows.map((row) => [row.kind, row.modelKey, row.isConfigured]),
    ).toEqual([
      ["search", "brave", true],
      ["search", "exa", false],
    ])
  })

  it("lists nothing before the instances are read", () => {
    expect(listInstanceRows("fetch", undefined)).toEqual([])
  })
})

describe("isSetInConfig", () => {
  const [configured] = listInstanceRows("search", {
    stored: [],
    config: [
      {
        name: "brave",
        kind: "search",
        provider: "brave",
        has_api_key: true,
        shadowed: false,
      },
    ],
  })

  it("holds a configured instance's rate written from its price_per_call", () => {
    expect(isSetInConfig(configured, "config")).toBe(true)
  })

  it("leaves a configured instance with an API rate editable", () => {
    expect(isSetInConfig(configured, "api")).toBe(false)
    expect(isSetInConfig(configured, "")).toBe(false)
  })

  it("never holds a gateway row, which no config entry prices", () => {
    expect(isSetInConfig(GATEWAY_ROWS.code_execution[0], "config")).toBe(false)
  })
})

describe("per-call rates", () => {
  it("reads and writes dollars per call against a per-million store", () => {
    expect(toPerCall(10_000)).toBe(0.01)
    expect(parsePerCallPrice("0.01")).toBe(10_000)
    expect(parsePerCallPrice(" 0.07 ")).toBe(70_000)
    expect(parsePerCallPrice("0")).toBe(0)
  })

  it.each(["", "1e3", "0x10", "-1", ".5", "abc"])("refuses %j", (text) => {
    expect(parsePerCallPrice(text)).toBeUndefined()
  })
})

describe("pairWithRates", () => {
  const rate = (modelKey: string, stored: number): RateRead => ({
    data: pricingResponse({
      model_key: modelKey,
      input_price_per_million: stored,
    }),
    isPending: false,
    isError: false,
  })
  const unpriced: RateRead = { data: null, isPending: false, isError: false }
  const [search, fetch] = GATEWAY_ROWS.web_search
  const instances = listInstanceRows("search", {
    stored: [],
    config: [
      {
        name: "brave",
        kind: "search",
        provider: "brave",
        has_api_key: true,
        shadowed: false,
      },
    ],
  })

  it("gives an instance its own tool's otari row to fall back to", () => {
    const entries = pairWithRates(
      [search, fetch, ...instances],
      [rate("otari:web_search", 10_000), unpriced, unpriced],
    )

    expect(entries[2].fallback?.data?.model_key).toBe("otari:web_search")
    expect(entries[0].fallback).toBeUndefined()
    // Web fetch has no rate and nothing to fall back to; brave inherits search's.
    expect(hasUnpricedRow(entries)).toBe(true)
    expect(hasUnpricedRow([entries[0], entries[2]])).toBe(false)
  })

  it("does not call an instance unpriced while the rate it inherits is unread", () => {
    const pending: RateRead = { isPending: true, isError: false }
    const failed: RateRead = { isPending: false, isError: true }
    for (const fallbackRead of [pending, failed]) {
      const entries = pairWithRates(
        [search, ...instances],
        [fallbackRead, unpriced],
      )
      expect(hasUnpricedRow([entries[1]])).toBe(false)
    }
  })

  it("does not call a row unpriced before its rate is read", () => {
    const pending: RateRead = { isPending: true, isError: false }
    expect(hasUnpricedRow(pairWithRates([search], [pending]))).toBe(false)
  })
})
