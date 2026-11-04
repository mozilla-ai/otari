import { describe, expect, it } from "vitest"

import type { AgentRecommender } from "@/client"

import {
  checkDraft,
  createBody,
  draftFrom,
  EMPTY_DRAFT,
  keyPlaceholder,
  parsePrice,
  patchBody,
  priceSummary,
  type RecommenderDraft,
} from "./recommenderDraft"

function stored(overrides: Partial<AgentRecommender> = {}): AgentRecommender {
  return {
    id: "11111111-1111-4111-8111-111111111111",
    provider: "openrouter",
    model: "stored-decider",
    api_base: null,
    api_key_last4: "1234",
    has_api_key: true,
    input_price_per_million: 20,
    output_price_per_million: 0,
    enabled: true,
    created_at: "2026-10-01T00:00:00Z",
    updated_at: null,
    ...overrides,
  }
}

function draft(overrides: Partial<RecommenderDraft> = {}): RecommenderDraft {
  return {
    ...EMPTY_DRAFT,
    provider: "openrouter",
    model: "stored-decider",
    apiKey: "sk-or-new",
    inputPrice: "20",
    ...overrides,
  }
}

describe("parsePrice", () => {
  it("reads a blank field as zero and a decimal as a number", () => {
    expect(parsePrice("  ")).toEqual({ ok: true, value: 0 })
    expect(parsePrice("0.15")).toEqual({ ok: true, value: 0.15 })
  })

  it("refuses a negative, a seventh decimal place and a word", () => {
    for (const text of ["-1", "0.1234567", "free"]) {
      expect(parsePrice(text).ok).toBe(false)
    }
  })
})

describe("priceSummary", () => {
  it("names both rates with no margin", () => {
    expect(
      priceSummary({
        input_price_per_million: 0.15,
        output_price_per_million: 0,
      }),
    ).toBe("$0.15 in, $0 out per 1M tokens")
  })
})

describe("draftFrom", () => {
  it("seeds every field but the key from a stored router", () => {
    expect(draftFrom(stored({ api_base: "https://proxy.example" }))).toEqual({
      provider: "openrouter",
      model: "stored-decider",
      apiBase: "https://proxy.example",
      apiKey: "",
      inputPrice: "20",
      outputPrice: "0",
    })
  })
})

describe("checkDraft", () => {
  it("passes a keyed provider with a price", () => {
    expect(checkDraft(draft(), undefined)).toEqual({})
  })

  it("needs a model, and a key for a provider that takes one", () => {
    expect(checkDraft(draft({ model: " ", apiKey: "" }), undefined)).toEqual({
      model: "Enter the decision model.",
      apiKey: "Enter the API key for this provider.",
    })
  })

  it("keeps the stored key only while the provider and the address stay the same", () => {
    const row = stored({ api_base: "https://proxy.example" })
    const same = draft({ apiKey: "", apiBase: "https://proxy.example" })
    expect(checkDraft(same, row)).toEqual({})
    expect(checkDraft({ ...same, provider: "typesafe" }, row).apiKey).toBe(
      "Enter the API key for this provider.",
    )
    expect(
      checkDraft({ ...same, apiBase: "https://other.example" }, row).apiKey,
    ).toBe("Enter the API key for this provider.")
    expect(checkDraft(same, stored({ has_api_key: false })).apiKey).toBe(
      "Enter the API key for this provider.",
    )
  })

  it("lets a self-hosted server go without a key but not without an address", () => {
    const server = draft({ provider: "llamacpp", apiKey: "", apiBase: "" })
    expect(checkDraft(server, undefined)).toEqual({
      apiBase: "Enter the server's address.",
    })
    expect(
      checkDraft(
        { ...server, apiBase: "http://decisions.internal" },
        undefined,
      ),
    ).toEqual({})
  })

  it("holds the address to https when a key travels to it, whatever the case", () => {
    expect(
      checkDraft(draft({ apiBase: "HTTP://proxy.example" }), undefined),
    ).toEqual({
      apiBase: "Use https: the API key is sent to this address.",
    })
    expect(
      checkDraft(draft({ apiBase: "ftp://proxy.example" }), undefined),
    ).toEqual({
      apiBase: "Enter an http or https address.",
    })
  })

  it("passes a free router with both rates at zero", () => {
    expect(
      checkDraft(draft({ inputPrice: "", outputPrice: "" }), undefined),
    ).toEqual({})
    expect(
      checkDraft(draft({ inputPrice: "0", outputPrice: "0" }), undefined),
    ).toEqual({})
    expect(
      checkDraft(draft({ inputPrice: "", outputPrice: "0.5" }), undefined),
    ).toEqual({})
    expect(
      checkDraft(draft({ inputPrice: "x" }), undefined).inputPrice,
    ).toContain("Enter an amount")
  })
})

describe("createBody and patchBody", () => {
  it("sends trimmed fields, leaves a blank address and key out, and carries no margin", () => {
    const body = createBody(
      draft({ model: " m ", apiKey: " k ", apiBase: " " }),
    )
    expect(body).toEqual({
      provider: "openrouter",
      model: "m",
      api_key: "k",
      input_price_per_million: 20,
      output_price_per_million: 0,
    })
    expect(body).not.toHaveProperty("margin_percent")
  })

  it("sends a blank address on a patch, so it goes back to the provider's own", () => {
    expect(patchBody(draft({ apiKey: "", apiBase: "" }))).toEqual({
      provider: "openrouter",
      model: "stored-decider",
      api_base: "",
      input_price_per_million: 20,
      output_price_per_million: 0,
    })
  })
})

describe("keyPlaceholder", () => {
  it("names the kept key by its tail, or by its presence when it has none", () => {
    expect(keyPlaceholder(draft({ apiKey: "" }), stored())).toBe(
      "Stored key ends in 1234. Leave blank to keep it.",
    )
    expect(
      keyPlaceholder(draft({ apiKey: "" }), stored({ api_key_last4: null })),
    ).toBe("A key is stored. Leave blank to keep it.")
    expect(keyPlaceholder(draft({ apiKey: "" }), undefined)).toBe("")
    expect(
      keyPlaceholder(draft({ provider: "typesafe", apiKey: "" }), stored()),
    ).toBe("")
  })
})
