import { describe, expect, it } from "vitest"

import type { CatalogModelSummary } from "@/client"
import {
  activeFilterCount,
  compareModels,
  EMPTY_FILTERS,
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
    selector: null,
    resolves_to: null,
    selectors: ["openai:model"],
    price_sources: [],
    unpriced_count: 0,
    discovered: true,
    min_input_price_per_million: null,
    min_output_price_per_million: null,
    ...overrides,
  }
}

const GLM = model({
  id: "z-ai/glm-5.3",
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
  selector: "z-ai/glm-5.3",
  resolves_to: "nebius:zai-org/GLM-5.3",
  selectors: [
    "fireworks:accounts/fireworks/models/glm-5p3",
    "nebius:zai-org/GLM-5.3",
  ],
  price_sources: ["defaults", "deployment"],
  min_input_price_per_million: 0.5,
  min_output_price_per_million: 2,
})
const KIMI = model({
  id: "moonshotai/kimi-k2.6",
  name: "Kimi K2.6",
  vendor: "Moonshot AI",
  input_modalities: ["image", "text"],
  context_window: 262_144,
  release_date: "2026-05-01",
  providers: ["nebius"],
  selector: null,
  resolves_to: null,
  selectors: ["nebius:moonshotai/Kimi-K2.6"],
  price_sources: ["defaults"],
  unpriced_count: 0,
  discovered: true,
  min_input_price_per_million: 0.6,
  min_output_price_per_million: 2.4,
})
const LOCAL = model({ id: "qwen3-32b", providers: ["home_lab"] })

const ANY = {
  ...EMPTY_FILTERS,
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
      filterModels([GLM, KIMI, LOCAL], { ...ANY, providers: ["fireworks"] }),
    ).toEqual([GLM])
    expect(
      filterModels([GLM, KIMI, LOCAL], { ...ANY, providers: ["nebius"] }),
    ).toEqual([GLM, KIMI])
  })

  it("treats an unknown vendor as its own bucket", () => {
    expect(filterModels([GLM, LOCAL], { ...ANY, vendors: [""] })).toEqual([
      LOCAL,
    ])
  })

  it("tests a capability against the model's own flags", () => {
    expect(
      filterModels([GLM, KIMI], { ...ANY, inputModalities: ["image"] }),
    ).toEqual([KIMI])
    expect(
      filterModels([GLM, KIMI], { ...ANY, capabilities: ["reasoning"] }),
    ).toEqual([GLM])
  })

  it("drops a model whose context is unknown from a context floor", () => {
    expect(
      filterModels([GLM, KIMI, LOCAL], { ...ANY, minContext: 250_000 }),
    ).toEqual([KIMI])
  })
})

describe("filterModels, the price and release filters", () => {
  const priced = [
    GLM,
    KIMI,
    model({
      id: "old-and-free",
      release_date: "2022-01-01",
      price_sources: [],
      unpriced_count: 1,
      discovered: false,
    }),
  ]
  const now = new Date("2026-09-10T00:00:00Z")

  it("searches selectors and provider instances too", () => {
    expect(
      filterModels(priced, { ...ANY, query: "accounts/fireworks" }).map(
        (m) => m.id,
      ),
    ).toEqual(["z-ai/glm-5.3"])
    expect(
      filterModels(priced, { ...ANY, query: "nebius" }).map((m) => m.id),
    ).toEqual(["z-ai/glm-5.3", "moonshotai/kimi-k2.6"])
  })

  it("narrows by where a price came from", () => {
    const ids = (pricing: string) =>
      filterModels(priced, { ...ANY, pricing }).map((m) => m.id)
    expect(ids("custom")).toEqual(["z-ai/glm-5.3"])
    expect(ids("default")).toEqual(["z-ai/glm-5.3", "moonshotai/kimi-k2.6"])
    expect(ids("priced")).toEqual(["z-ai/glm-5.3", "moonshotai/kimi-k2.6"])
    expect(ids("unpriced")).toEqual(["old-and-free"])
  })

  it("narrows by whether a provider discovered the model", () => {
    expect(
      filterModels(priced, { ...ANY, source: "custom" }).map((m) => m.id),
    ).toEqual(["old-and-free"])
    expect(filterModels(priced, { ...ANY, source: "discovered" })).toHaveLength(
      2,
    )
  })

  it("caps the cheapest input rate, dropping the unpriced", () => {
    expect(
      filterModels(priced, { ...ANY, maxInput: 0.55 }).map((m) => m.id),
    ).toEqual(["z-ai/glm-5.3"])
  })

  it("keeps only a release inside the window, and none with no date", () => {
    expect(
      filterModels(priced, { ...ANY, releasedWithinDays: 365 }, now).map(
        (m) => m.id,
      ),
    ).toEqual(["z-ai/glm-5.3", "moonshotai/kimi-k2.6"])
    expect(
      filterModels(priced, { ...ANY, releasedWithinDays: 3 * 365 }, now).map(
        (m) => m.id,
      ),
    ).toEqual(["z-ai/glm-5.3", "moonshotai/kimi-k2.6"])
    expect(
      filterModels(priced, { ...ANY, releasedWithinDays: 5 * 365 }, now),
    ).toHaveLength(3)
  })
})

describe("compareModels", () => {
  it("puts an unpriced model last whichever way the price sorts", () => {
    const asc = [LOCAL, KIMI, GLM].sort(compareModels("input", "asc"))
    expect(asc.map((m) => m.id)).toEqual([
      "z-ai/glm-5.3",
      "moonshotai/kimi-k2.6",
      "qwen3-32b",
    ])
    const desc = [LOCAL, KIMI, GLM].sort(compareModels("input", "desc"))
    expect(desc.map((m) => m.id)).toEqual([
      "moonshotai/kimi-k2.6",
      "z-ai/glm-5.3",
      "qwen3-32b",
    ])
  })

  it("sorts release dates lexically, which is chronological for ISO dates", () => {
    const newest = [GLM, KIMI, LOCAL].sort(compareModels("released", "desc"))
    expect(newest.map((m) => m.id)).toEqual([
      "z-ai/glm-5.3",
      "moonshotai/kimi-k2.6",
      "qwen3-32b",
    ])
  })
})

describe("options", () => {
  it("lists vendors with the unknown bucket named", () => {
    expect(vendorOptions([GLM, LOCAL])).toEqual([
      { value: "", label: "Unknown vendor" },
      { value: "Z.ai", label: "Z.ai" },
    ])
  })

  it("lists every provider instance once", () => {
    expect(providerOptions([GLM, KIMI]).map((o) => o.value)).toEqual([
      "fireworks",
      "nebius",
    ])
  })

  it("narrows by what a model produces", () => {
    const image = model({ id: "painter", output_modalities: ["image"] })
    expect(
      filterModels([GLM, KIMI, image], { ...ANY, outputModalities: ["image"] }),
    ).toEqual([image])
  })

  it("counts the choices in force, and not the search", () => {
    expect(activeFilterCount(ANY)).toBe(0)
    expect(
      activeFilterCount({
        ...ANY,
        query: "glm",
        outputModalities: ["text"],
        providers: ["nebius", "fireworks"],
        minContext: 32_000,
        pricing: "custom",
      }),
    ).toBe(5)
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
