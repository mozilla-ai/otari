import { describe, expect, it } from "vitest"

import type { CatalogModelSummary } from "@/client"
import {
  activeFilterCount,
  CATALOG_URL_DEFAULTS,
  type CatalogListState,
  type CatalogUrlKey,
  catalogListFromUrl,
  catalogListToUrl,
  catalogRequest,
  catalogUrlChanges,
  credentialLabel,
  EMPTY_CATALOG_LIST,
  EMPTY_FILTERS,
  makerKeyOf,
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
const LOCAL = model({ id: "qwen3-32b", providers: ["home_lab"] })

const ANY = {
  ...EMPTY_FILTERS,
}

describe("catalogRequest", () => {
  it("maps the complete filter state and page to the API", () => {
    expect(
      catalogRequest(
        {
          ...ANY,
          query: "  moonshot  ",
          providers: ["nebius", "fireworks"],
          vendors: [""],
          inputModalities: ["text", "image"],
          outputModalities: ["audio"],
          capabilities: ["reasoning", "tool_call"],
          minContext: 128000,
          maxInput: 3,
          pricing: "custom",
          source: "discovered",
          releasedWithinDays: 365,
        },
        "input",
        "desc",
        2,
        25,
      ),
    ).toEqual({
      search: "moonshot",
      provider: ["fireworks", "nebius"],
      vendor: [""],
      input_modality: ["image", "text"],
      output_modality: ["audio"],
      capability: ["reasoning", "tool_call"],
      min_context: 128000,
      max_input: 3,
      pricing: "custom",
      source: "discovered",
      released_within_days: 365,
      sort: "input",
      direction: "desc",
      skip: 50,
      limit: 25,
      include_facets: true,
    })
  })

  it("omits inactive numeric filters and blank search", () => {
    const params = catalogRequest(ANY, "name", "asc", 0, 25)
    expect(params.search).toBeUndefined()
    expect(params.min_context).toBeUndefined()
    expect(params.max_input).toBeUndefined()
    expect(params.released_within_days).toBeUndefined()
  })
})

// Reads a query the way `useUrlState` does: a missing key is its default, and a
// blank value is no value.
function readQuery(params: URLSearchParams) {
  return {
    get: (key: CatalogUrlKey) => params.get(key) ?? CATALOG_URL_DEFAULTS[key],
    getAll: (key: CatalogUrlKey) => params.getAll(key).filter(Boolean),
    getNumber: (key: CatalogUrlKey) =>
      Number.parseInt(params.get(key) ?? CATALOG_URL_DEFAULTS[key], 10),
  }
}

function writeQuery(list: CatalogListState): URLSearchParams {
  const params = new URLSearchParams()
  for (const [key, value] of Object.entries(catalogListToUrl(list))) {
    for (const one of Array.isArray(value) ? value : [value]) {
      params.append(key, String(one))
    }
  }
  return params
}

describe("the list's URL", () => {
  it("round-trips every filter, both sorts and the page", () => {
    const list: CatalogListState = {
      filters: {
        query: "glm",
        inputModalities: ["image"],
        outputModalities: ["text"],
        providers: ["fireworks", "nebius"],
        vendors: ["", "Z.ai"],
        capabilities: ["reasoning"],
        minContext: 128_000,
        maxInput: 3,
        pricing: "priced",
        source: "discovered",
        releasedWithinDays: 365,
      },
      sort: "name",
      tableSort: { column: "input", direction: "desc" },
      page: 2,
      pageSize: 50,
    }
    expect(catalogListFromUrl(readQuery(writeQuery(list)))).toEqual(list)
  })

  it("names only the params that changed", () => {
    const from = {
      ...EMPTY_CATALOG_LIST,
      filters: { ...EMPTY_FILTERS, query: "gl" },
    }
    const to = {
      ...from,
      filters: { ...from.filters, capabilities: ["reasoning"] },
      page: 0,
    }
    expect(catalogUrlChanges(from, to, { q: "glm" })).toEqual({
      capability: ["reasoning"],
    })
  })

  it("adds and removes only this change's values in a repeated param", () => {
    // The URL already holds a change the rendered list has not caught up with.
    const seen = {
      ...EMPTY_CATALOG_LIST,
      filters: { ...EMPTY_FILTERS, inputModalities: ["pdf"] },
    }
    const latest = { input: ["pdf", "text"] }
    const addImage = {
      ...seen,
      filters: { ...seen.filters, inputModalities: ["pdf", "image"] },
    }
    expect(catalogUrlChanges(seen, addImage, latest)).toEqual({
      input: ["pdf", "text", "image"],
    })
    const dropPdf = {
      ...seen,
      filters: { ...seen.filters, inputModalities: [] },
    }
    expect(catalogUrlChanges(seen, dropPdf, latest)).toEqual({
      input: ["text"],
    })
  })
})

describe("makerKeyOf", () => {
  it("reads the vendor slug off the catalog id", () => {
    // The gateway builds the id as `vendor_slug(vendor)/slug`, so the prefix is
    // the normalized vendor. Taking it beats reimplementing those rules here:
    // `Z.ai` normalizes to `z-ai`, which no obvious slug function would guess.
    expect(makerKeyOf(GLM)).toBe("z-ai")
  })

  it("has no key for a model whose maker is unknown", () => {
    // No vendor means the gateway added no prefix, so there is nothing to read.
    expect(makerKeyOf(LOCAL)).toBeUndefined()
  })

  it("needs both halves before it trusts a prefix", () => {
    // A vendor with no separator in the id is not a shape the gateway produces,
    // so it yields nothing rather than a guess at where the slug ends.
    expect(makerKeyOf({ id: "glm-5.3", vendor: "Z.ai" })).toBeUndefined()
    // And a slash with no vendor is a model whose name simply has one in it.
    expect(makerKeyOf({ id: "org/model", vendor: null })).toBeUndefined()
  })
})

describe("options", () => {
  it("lists vendors with the unknown bucket named", () => {
    // `markKey` is the vendor slug a mark is keyed on, carried alongside because
    // the filter matches on the display string and only a row knows both.
    expect(
      vendorOptions([
        { value: "Z.ai", vendor_slug: "z-ai" },
        { value: "", vendor_slug: null },
      ]),
    ).toEqual([
      { value: "", label: "Unknown vendor", markKey: undefined },
      { value: "Z.ai", label: "Z.ai", markKey: "z-ai" },
    ])
  })

  it("lists every provider instance once", () => {
    expect(
      providerOptions(["fireworks", "nebius"]).map((o) => o.value),
    ).toEqual(["fireworks", "nebius"])
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

describe("credentialLabel", () => {
  it("names whose key serves the offering", () => {
    expect(credentialLabel("organization")).toBe("org key")
    expect(credentialLabel("hosted")).toBe("hosted")
    expect(credentialLabel("deployment")).toBe("deployment")
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
