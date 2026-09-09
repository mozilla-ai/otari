import { describe, expect, it } from "vitest"

import type { CatalogModelSummary } from "@/client"
import {
  compareModels,
  filterModels,
  priceSourceLabel,
  providerOptions,
  vendorOptions,
} from "@/features/models/catalog"

function model(
  overrides: Partial<CatalogModelSummary> & Pick<CatalogModelSummary, "id">,
): CatalogModelSummary {
  return {
    name: overrides.id,
    vendor: null,
    family: null,
    capabilities: {
      reasoning: false,
      tool_call: false,
      structured_output: false,
      attachment: false,
      temperature: false,
    },
    input_modalities: ["text"],
    output_modalities: ["text"],
    context_window: null,
    max_output_tokens: null,
    release_date: null,
    knowledge_cutoff: null,
    open_weights: false,
    deprecated: false,
    offering_count: 1,
    provider_count: 1,
    providers: ["openai"],
    min_input_price_per_million: null,
    min_output_price_per_million: null,
    ...overrides,
  }
}

const GLM = model({
  id: "glm-5-3",
  name: "GLM-5.3",
  vendor: "Z.ai",
  capabilities: {
    reasoning: true,
    tool_call: true,
    structured_output: false,
    attachment: false,
    temperature: true,
  },
  context_window: 200_000,
  release_date: "2026-07-01",
  providers: ["fireworks", "nebius"],
  provider_count: 2,
  offering_count: 2,
  min_input_price_per_million: 0.5,
  min_output_price_per_million: 2,
})
const KIMI = model({
  id: "kimi-k2-6",
  name: "Kimi K2.6",
  vendor: "Moonshot AI",
  input_modalities: ["image", "text"],
  context_window: 262_144,
  release_date: "2026-05-01",
  providers: ["nebius"],
  min_input_price_per_million: 0.6,
  min_output_price_per_million: 2.4,
})
const LOCAL = model({ id: "qwen3-32b", providers: ["home_lab"] })

const ANY = {
  query: "",
  vendor: "all",
  provider: "all",
  capability: "all",
  minContext: 0,
}

describe("filterModels", () => {
  it("searches the name, the vendor and the id", () => {
    expect(filterModels([GLM, KIMI], { ...ANY, query: "moonshot" })).toEqual([
      KIMI,
    ])
    expect(filterModels([GLM, KIMI], { ...ANY, query: "glm-5" })).toEqual([GLM])
  })

  it("narrows by provider instance, the way the Providers page links here", () => {
    expect(
      filterModels([GLM, KIMI, LOCAL], { ...ANY, provider: "fireworks" }),
    ).toEqual([GLM])
    expect(
      filterModels([GLM, KIMI, LOCAL], { ...ANY, provider: "nebius" }),
    ).toEqual([GLM, KIMI])
  })

  it("treats an unknown vendor as its own bucket", () => {
    expect(filterModels([GLM, LOCAL], { ...ANY, vendor: "" })).toEqual([LOCAL])
  })

  it("tests a capability against the model's own flags", () => {
    expect(filterModels([GLM, KIMI], { ...ANY, capability: "vision" })).toEqual(
      [KIMI],
    )
    expect(
      filterModels([GLM, KIMI], { ...ANY, capability: "reasoning" }),
    ).toEqual([GLM])
  })

  it("drops a model whose context is unknown from a context floor", () => {
    expect(
      filterModels([GLM, KIMI, LOCAL], { ...ANY, minContext: 250_000 }),
    ).toEqual([KIMI])
  })
})

describe("compareModels", () => {
  it("puts an unpriced model last whichever way the price sorts", () => {
    const asc = [LOCAL, KIMI, GLM].sort(compareModels("input", "asc"))
    expect(asc.map((m) => m.id)).toEqual(["glm-5-3", "kimi-k2-6", "qwen3-32b"])
    const desc = [LOCAL, KIMI, GLM].sort(compareModels("input", "desc"))
    expect(desc.map((m) => m.id)).toEqual(["kimi-k2-6", "glm-5-3", "qwen3-32b"])
  })

  it("sorts release dates lexically, which is chronological for ISO dates", () => {
    const newest = [GLM, KIMI, LOCAL].sort(compareModels("released", "desc"))
    expect(newest.map((m) => m.id)).toEqual([
      "glm-5-3",
      "kimi-k2-6",
      "qwen3-32b",
    ])
  })
})

describe("options", () => {
  it("lists vendors with the unknown bucket named", () => {
    expect(vendorOptions([GLM, LOCAL])).toEqual([
      { value: "all", label: "All vendors" },
      { value: "", label: "Unknown vendor" },
      { value: "Z.ai", label: "Z.ai" },
    ])
  })

  it("lists every provider instance once", () => {
    expect(providerOptions([GLM, KIMI]).map((o) => o.value)).toEqual([
      "all",
      "fireworks",
      "nebius",
    ])
  })
})

describe("priceSourceLabel", () => {
  it("names the rung, and the absence of one", () => {
    expect(priceSourceLabel("organization")).toBe("your rate")
    expect(priceSourceLabel("deployment")).toBe("custom")
    expect(priceSourceLabel("defaults")).toBe("default")
    expect(priceSourceLabel(null)).toBe("not priced")
  })
})
