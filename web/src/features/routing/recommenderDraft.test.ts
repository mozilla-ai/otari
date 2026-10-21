import { describe, expect, it } from "vitest"

import type { AgentRecommender, DecisionProviderOption } from "@/client"

import {
  checkDraft,
  createBody,
  draftFrom,
  EMPTY_DRAFT,
  keyPlaceholder,
  parsePrice,
  patchBody,
  priceSummary,
  providerLabel,
  providerOptions,
} from "./recommenderDraft"

const providers: DecisionProviderOption[] = [
  { provider: "typesafe", self_hosted: false },
  { provider: "openrouter", self_hosted: false },
  { provider: "llamacpp", self_hosted: true },
]

const stored: AgentRecommender = {
  id: "7f1c2a9e-0000-4000-8000-000000000001",
  provider: "typesafe",
  model: "model-a",
  api_base: null,
  api_key_last4: "1234",
  has_api_key: true,
  input_price_per_million: 0.15,
  output_price_per_million: 0,
  enabled: true,
  created_at: "2026-10-08T00:00:00Z",
  updated_at: null,
}

describe("parsePrice", () => {
  it("accepts a decimal at the ledger's precision, and blank as zero", () => {
    expect(parsePrice(" 0.000125 ")).toEqual({ ok: true, value: 0.000125 })
    expect(parsePrice("1")).toEqual({ ok: true, value: 1 })
    expect(parsePrice(" ")).toEqual({ ok: true, value: 0 })
  })

  it("refuses a negative amount and what the ledger would round", () => {
    expect(parsePrice("0.0000001").ok).toBe(false)
    expect(parsePrice("-1").ok).toBe(false)
    expect(parsePrice("1e-3").ok).toBe(false)
  })

  it("refuses more whole digits than the store keeps", () => {
    expect(parsePrice("123456789012").ok).toBe(true)
    expect(parsePrice("1234567890123").ok).toBe(false)
  })
})

describe("providers", () => {
  it("names the providers the server lists, and one it adds later by its own name", () => {
    expect(
      providerOptions([...providers, { provider: "next", self_hosted: false }]),
    ).toEqual([
      { value: "typesafe", label: "TypeSafe" },
      { value: "openrouter", label: "OpenRouter" },
      { value: "llamacpp", label: "llama.cpp server" },
      { value: "next", label: "next" },
    ])
    expect(providerLabel("openrouter")).toBe("OpenRouter")
  })
})

describe("draftFrom", () => {
  it("starts from the stored recommender, without its key", () => {
    expect(draftFrom(stored)).toEqual({
      provider: "typesafe",
      model: "model-a",
      apiBase: "",
      apiKey: "",
      inputPrice: "0.15",
      outputPrice: "0",
    })
  })
})

describe("checkDraft", () => {
  it("asks for a key unless one is stored for the same provider", () => {
    const draft = draftFrom(stored)
    expect(checkDraft(draft, stored, providers)).toEqual({})
    expect(
      checkDraft({ ...draft, provider: "openrouter" }, stored, providers),
    ).toEqual({ apiKey: "Enter the API key for this provider." })
  })

  it("asks a new recommender for its own key", () => {
    expect(checkDraft(EMPTY_DRAFT, undefined, providers).apiKey).toBe(
      "Enter the API key for this provider.",
    )
  })

  it("lets a self-hosted server go without a key, but not without its address", () => {
    const draft = {
      ...EMPTY_DRAFT,
      provider: "llamacpp",
      model: "m",
      inputPrice: "0.1",
    }
    expect(checkDraft(draft, undefined, providers)).toEqual({
      apiBase: "Enter the server's address.",
    })
    expect(
      checkDraft(
        { ...draft, apiBase: "http://decisions:8080" },
        undefined,
        providers,
      ),
    ).toEqual({})
  })

  it("asks for the key again when the address changes, since the key goes there", () => {
    const draft = { ...draftFrom(stored), apiBase: "https://elsewhere.example" }
    expect(checkDraft(draft, stored, providers)).toEqual({
      apiKey: "Enter the API key for this provider.",
    })
    expect(keyPlaceholder(draft, stored)).toBe("")
  })

  it("refuses a plain http address when a key travels to it, in any case", () => {
    const draft = { ...draftFrom(stored), apiKey: "new-key" }
    for (const apiBase of ["http://proxy.example", "HTTP://proxy.example"]) {
      expect(checkDraft({ ...draft, apiBase }, stored, providers).apiBase).toBe(
        "Use https: the API key is sent to this address.",
      )
    }
  })

  it("refuses an address that is not http or https", () => {
    const draft = { ...draftFrom(stored), apiBase: "ftp://proxy.example" }
    expect(checkDraft(draft, stored, providers).apiBase).toBe(
      "Enter an http or https address.",
    )
  })

  it("asks for a model", () => {
    const draft = { ...draftFrom(stored), model: " " }
    expect(checkDraft(draft, stored, providers)).toEqual({
      model: "Enter the decision model.",
    })
  })

  it("takes either token price", () => {
    const draft = { ...draftFrom(stored), inputPrice: "", outputPrice: "0.6" }
    expect(checkDraft(draft, stored, providers)).toEqual({})
  })

  it("refuses a recommendation that would be free", () => {
    const draft = { ...draftFrom(stored), inputPrice: "0" }
    expect(checkDraft(draft, stored, providers)).toEqual({
      inputPrice: "Set an input or output token price above zero.",
    })
  })

  it("names each price it cannot read", () => {
    const draft = { ...draftFrom(stored), outputPrice: "-1" }
    expect(Object.keys(checkDraft(draft, stored, providers))).toEqual([
      "outputPrice",
    ])
  })
})

describe("createBody", () => {
  it("sends what was typed, trimmed, a blank price as zero, and lands switched off", () => {
    const draft = {
      ...EMPTY_DRAFT,
      model: " model-b ",
      apiBase: " https://proxy.example ",
      apiKey: " new-key ",
      inputPrice: " 0.2 ",
    }
    expect(createBody(draft)).toEqual({
      provider: "typesafe",
      model: "model-b",
      api_base: "https://proxy.example",
      api_key: "new-key",
      input_price_per_million: 0.2,
      output_price_per_million: 0,
      enabled: false,
    })
  })

  it("leaves a blank address out, for the provider's own", () => {
    expect(createBody({ ...EMPTY_DRAFT, model: "m" })).not.toHaveProperty(
      "api_base",
    )
  })
})

describe("patchBody", () => {
  it("leaves the key out when none was typed, so the stored one is kept, and switches the router off", () => {
    expect(patchBody(draftFrom(stored))).toEqual({
      provider: "typesafe",
      model: "model-a",
      api_base: "",
      input_price_per_million: 0.15,
      output_price_per_million: 0,
      enabled: false,
    })
  })

  it("sends a new key when one was typed", () => {
    expect(patchBody({ ...draftFrom(stored), apiKey: " k " }).api_key).toBe("k")
  })
})

describe("keyPlaceholder", () => {
  it("names the stored key's tail when it would be kept", () => {
    expect(keyPlaceholder(draftFrom(stored), stored)).toBe(
      "Stored key ends in 1234. Leave blank to keep it.",
    )
    expect(
      keyPlaceholder({ ...draftFrom(stored), provider: "openrouter" }, stored),
    ).toBe("")
    expect(keyPlaceholder(EMPTY_DRAFT, undefined)).toBe("")
  })

  it("says a key is kept when it is too short to show its tail", () => {
    const short = { ...stored, api_key_last4: null }
    expect(keyPlaceholder(draftFrom(short), short)).toBe(
      "A key is stored. Leave blank to keep it.",
    )
    expect(
      checkDraft(draftFrom(short), short, providers).apiKey,
    ).toBeUndefined()
  })
})

describe("priceSummary", () => {
  it("names the token prices and nothing on top", () => {
    expect(priceSummary(stored)).toBe("$0.15 in, $0.00 out per 1M tokens")
  })
})
