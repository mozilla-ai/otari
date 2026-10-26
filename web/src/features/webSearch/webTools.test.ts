import { describe, expect, it } from "vitest"

import type {
  SearchProviderInfo,
  SearchProviderOption,
  SearchToolsResponse,
} from "@/client"
import type { RateRead } from "@/features/toolPricing/toolRates"
import { pricingResponse } from "@/tests/fixtures"

import {
  buildOptions,
  hasUnpricedCalls,
  isNoSearch,
  listWebTools,
  nameProblem,
  perCallText,
  priceOf,
  rateOnly,
  sameOptions,
  servesTheModel,
  splitOptions,
  suggestName,
  testOutcome,
} from "./webTools"

const read = (perMillion?: number, origin = "operator"): RateRead => ({
  data:
    perMillion === undefined
      ? null
      : pricingResponse({
          model_key: "tool",
          input_price_per_million: perMillion,
          origin,
        }),
  isPending: false,
  isError: false,
})
const pending: RateRead = { isPending: true, isError: false }

const option = (
  name: string,
  type: SearchProviderOption["type"],
  extra: Partial<SearchProviderOption> = {},
): SearchProviderOption => ({
  name,
  type,
  description: "",
  operator_only: false,
  ...extra,
})

const provider = (
  overrides: Partial<SearchProviderInfo> = {},
): SearchProviderInfo => ({
  id: "exa",
  kind: "search",
  requires_api_key: true,
  requires_api_base: false,
  tier: "production",
  instances: [],
  options: [
    option("type", "string", { enum: ["auto", "fast"] }),
    option("numResults", "integer"),
    option("moderation", "boolean"),
    option("contents", "object"),
  ],
  ...overrides,
})

describe("listWebTools", () => {
  const tools: SearchToolsResponse = {
    stored: [
      {
        name: "exa",
        kind: "search",
        provider: "exa",
        last4: "a1b2",
        options: { type: "fast" },
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
        name: "mojeek",
        kind: "search",
        provider: "mojeek",
        has_api_key: true,
        shadowed: false,
      },
    ],
  }

  it("lists the config.yml tools a stored one does not shadow, then the stored", () => {
    const rows = listWebTools("search", tools)
    expect(rows.map((row) => [row.name, row.source])).toEqual([
      ["mojeek", "config"],
      ["exa", "stored"],
    ])
    expect(rows[1]).toMatchObject({
      last4: "a1b2",
      hasKey: true,
      overridesConfig: true,
      options: { type: "fast" },
    })
  })

  it("marks the built-in fetcher apart from the config.yml tools", () => {
    const rows = listWebTools("fetch", {
      stored: [],
      config: [
        {
          name: "builtin_fetch",
          kind: "fetch",
          provider: "builtin",
          has_api_key: false,
          shadowed: false,
        },
      ],
    })
    expect(rows[0].source).toBe("builtin")
  })

  it("lists nothing before the tools are read", () => {
    expect(listWebTools("search", undefined)).toEqual([])
  })
})

describe("priceOf", () => {
  const stored = { kind: "search", source: "stored" } as const

  it("takes the tool's own rate first", () => {
    expect(priceOf(stored, read(5000), read(10000))).toEqual({
      state: "own",
      perMillion: 5000,
      fromConfig: false,
    })
  })

  it("marks a config.yml tool's rate written from its price_per_call", () => {
    const configured = { kind: "search", source: "config" } as const
    expect(priceOf(configured, read(3000, "config"), read(10000))).toEqual({
      state: "own",
      perMillion: 3000,
      fromConfig: true,
    })
    // A stored tool's rate is never config.yml's to change.
    expect(priceOf(stored, read(3000, "config"), read(10000))).toMatchObject({
      fromConfig: false,
    })
  })

  it("leaves a search tool's search API calls unpriced on the fallback alone", () => {
    const price = priceOf(stored, read(), read(10000))
    expect(price).toEqual({
      state: "fallback",
      perMillion: 10000,
      fallbackKey: "otari:web_search",
      direct: "unpriced",
    })
    expect(hasUnpricedCalls(price)).toBe(true)
  })

  it("prices a fetch tool fully on the fallback, since it has no other door", () => {
    const price = priceOf(
      { kind: "fetch", source: "stored" },
      read(),
      read(1000),
    )
    expect(price).toMatchObject({ state: "fallback", direct: "none" })
    expect(hasUnpricedCalls(price)).toBe(false)
  })

  it("is not unpriced until the rate it would inherit is read too", () => {
    expect(priceOf(stored, read(), pending)).toEqual({ state: "pending" })
    expect(priceOf(stored, read(), read())).toEqual({ state: "unpriced" })
  })

  it("never marks a fallback rate as config.yml's", () => {
    expect(rateOnly(read(10000, "config"))).toMatchObject({
      state: "own",
      fromConfig: false,
    })
    expect(rateOnly(read())).toEqual({ state: "unpriced" })
  })
})

describe("names", () => {
  it("suggests the provider, or <provider>-fetch beside a search tool of that name", () => {
    expect(suggestName("search", "exa", [])).toBe("exa")
    expect(suggestName("fetch", "exa", ["exa"])).toBe("exa-fetch")
    expect(suggestName("search", "exa", ["exa", "exa-2"])).toBe("exa-3")
    expect(suggestName("search", "", [])).toBe("")
  })

  it("refuses what the API refuses", () => {
    expect(nameProblem("a:b", [])).toMatch(/':' or '\/'/)
    expect(nameProblem("a/b", [])).toMatch(/':' or '\/'/)
    expect(nameProblem("Builtin_Fetch", [])).toMatch(/reserved/)
    expect(nameProblem("NONE", [])).toMatch(/reserved/)
    expect(nameProblem("*", [])).toMatch(/reserved/)
    expect(nameProblem("exa", ["exa"])).toMatch(/already has this name/)
    expect(nameProblem("exa", [])).toBe("")
  })
})

describe("options", () => {
  it("splits stored options into typed fields and the JSON box", () => {
    expect(
      splitOptions(provider(), {
        type: "fast",
        numResults: 5,
        moderation: false,
        contents: { text: true },
        unknown: "kept",
      }),
    ).toEqual({
      typed: { type: "fast", numResults: "5", moderation: "false" },
      extraJson: JSON.stringify(
        { contents: { text: true }, unknown: "kept" },
        null,
        2,
      ),
    })
  })

  it("sends filled fields over the JSON box, and leaves blank ones out", () => {
    expect(
      buildOptions(
        provider(),
        { type: "auto", numResults: "7", moderation: "" },
        '{"type": "fast", "contents": {"text": true}}',
      ),
    ).toEqual({
      ok: true,
      options: { type: "auto", numResults: 7, contents: { text: true } },
    })
  })

  it("names the field or the JSON that cannot be sent", () => {
    const built = buildOptions(provider(), { numResults: "seven" }, "[1]")
    expect(built).toEqual({
      ok: false,
      fieldErrors: { numResults: "A whole number." },
      jsonError: expect.stringMatching(/JSON object/),
    })
    expect(buildOptions(provider(), {}, "{nope")).toMatchObject({
      ok: false,
      jsonError: "Not valid JSON.",
    })
  })

  it("sends a provider with no schema its options as written", () => {
    expect(
      buildOptions(provider({ options: null }), {}, '{"engines": "x"}'),
    ).toEqual({ ok: true, options: { engines: "x" } })
  })
})

describe("servesTheModel", () => {
  it("is false for a provider the search library has no adapter for", () => {
    expect(servesTheModel(provider({ id: "searxng", tier: null }))).toBe(false)
    expect(servesTheModel(provider())).toBe(true)
  })
})

describe("sameOptions", () => {
  it("ignores key order and unset values", () => {
    expect(
      sameOptions({ a: 1, b: { y: 2, x: 1 } }, { b: { x: 1, y: 2 }, a: 1 }),
    ).toBe(true)
    expect(sameOptions({ a: 1, c: null }, { a: 1 })).toBe(true)
    expect(sameOptions({ a: [1, 2] }, { a: [2, 1] })).toBe(false)
    expect(sameOptions({ a: 1 }, { a: 2 })).toBe(false)
  })
})

describe("perCallText", () => {
  it("writes a rate out in full, never in exponent form", () => {
    expect(perCallText(0.5)).toBe("0.0000005")
    expect(perCallText(5000)).toBe("0.005")
    expect(perCallText(1_000_000)).toBe("1")
    expect(perCallText(0)).toBe("0")
  })
})

describe("isNoSearch", () => {
  it("matches none in any case, as the gateway does", () => {
    expect(isNoSearch("None")).toBe(true)
    expect(isNoSearch(" NONE ")).toBe(true)
    expect(isNoSearch("exa")).toBe(false)
    expect(isNoSearch("")).toBe(false)
  })
})

describe("testOutcome", () => {
  it("says what a test answered, counting hits in the singular too", () => {
    expect(testOutcome("search", { ok: true, hits: 1 })).toBe("Worked: 1 hit.")
    expect(testOutcome("fetch", { ok: true, characters: 120 })).toBe(
      "Worked: 120 characters of page text.",
    )
    expect(testOutcome("search", { ok: false, error: "http_error" })).toBe(
      "Failed: http_error.",
    )
  })
})
