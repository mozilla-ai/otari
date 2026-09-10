import type {
  CatalogCapabilities,
  CatalogModelSummary,
  CatalogOffering,
  ModelPricingInfo,
} from "@/client"

// What the list can be narrowed by, and how a row is sorted. Pure, so the page
// stays a composition and these can be pinned on their own.

export const MODALITY_LABELS: Record<string, string> = {
  text: "Text",
  image: "Image",
  audio: "Audio",
  video: "Video",
  pdf: "PDF",
}

export const CAPABILITY_LABELS: {
  key: keyof CatalogCapabilities
  label: string
}[] = [
  { key: "reasoning", label: "Reasoning" },
  { key: "tool_call", label: "Tool calling" },
  { key: "structured_output", label: "Structured output" },
  { key: "attachment", label: "Attachments" },
  { key: "temperature", label: "Temperature" },
]

// The capability filter tests the model's own flags and modalities, so a pick
// narrows to models that actually report it rather than to a provider's coarse
// claim about everything it serves.
export const CAPABILITY_FILTERS: {
  value: string
  label: string
  test: (model: CatalogModelSummary) => boolean
}[] = [
  {
    value: "tool_call",
    label: "Tool calling",
    test: (m) => m.capabilities.tool_call,
  },
  {
    value: "reasoning",
    label: "Reasoning",
    test: (m) => m.capabilities.reasoning,
  },
  {
    value: "structured_output",
    label: "Structured output",
    test: (m) => m.capabilities.structured_output,
  },
  {
    value: "attachment",
    label: "Attachments",
    test: (m) => m.capabilities.attachment,
  },
  {
    value: "open_weights",
    label: "Open weights",
    test: (m) => m.open_weights,
  },
]

/** The modalities the rail offers, in the order they are listed. */
export const MODALITIES = ["text", "image", "pdf", "audio", "video"]

export const CONTEXT_OPTIONS = [
  { value: "0", label: "Any context" },
  { value: "32000", label: "≥ 32K" },
  { value: "128000", label: "≥ 128K" },
  { value: "200000", label: "≥ 200K" },
  { value: "1000000", label: "≥ 1M" },
]

// Which price list a model's offerings draw on. "custom" is a rate somebody
// here set, the deployment's or the organization's; "default" is genai-prices.
export const PRICING_OPTIONS = [
  { value: "all", label: "Any pricing" },
  { value: "custom", label: "Custom price" },
  { value: "default", label: "Default price" },
  { value: "priced", label: "Priced" },
  { value: "unpriced", label: "Unpriced" },
]

export const SOURCE_OPTIONS = [
  { value: "all", label: "Any source" },
  { value: "discovered", label: "Discovered" },
  { value: "custom", label: "Custom (not discovered)" },
]

/** A ceiling on the cheapest offering's input rate, in $ per million. */
export const PRICE_OPTIONS = [
  { value: "0", label: "Any price" },
  { value: "1", label: "≤ $1 / 1M in" },
  { value: "3", label: "≤ $3 / 1M in" },
  { value: "10", label: "≤ $10 / 1M in" },
  { value: "30", label: "≤ $30 / 1M in" },
]

// Newness windows in days back from today. A model with no known release
// date is excluded once a window is active.
export const RELEASE_OPTIONS = [
  { value: "0", label: "Any release date" },
  { value: "365", label: "Past year" },
  { value: "730", label: "Past 2 years" },
  { value: "1095", label: "Past 3 years" },
]

// The request size a price is compared at. Not a filter on the rows but on
// the numbers: a tiered offering is cheap at 8K and not at 500K. The list is
// re-read from the gateway at the chosen size; a model page reprices its
// offerings on the spot from the tiers it already holds.
export const COMPARE_AT_OPTIONS = [
  { value: "0", label: "Base prices" },
  { value: "8000", label: "Compare at 8K" },
  { value: "128000", label: "Compare at 128K" },
  { value: "200000", label: "Compare at 200K" },
  { value: "500000", label: "Compare at 500K" },
  { value: "1000000", label: "Compare at 1M" },
]

const DAY_MS = 24 * 60 * 60 * 1000

export interface CatalogFilters {
  query: string
  inputModalities: string[]
  outputModalities: string[]
  /** Provider instances; empty for any. The Providers page links here with one. */
  providers: string[]
  vendors: string[]
  /** Values of `CAPABILITY_FILTERS`; every one picked must hold. */
  capabilities: string[]
  minContext: number
  /** A ceiling on the cheapest input rate; 0 for none. */
  maxInput: number
  pricing: string
  source: string
  /** Days back from `now` a release must fall within; 0 for any. */
  releasedWithinDays: number
}

export const EMPTY_FILTERS: CatalogFilters = {
  query: "",
  inputModalities: [],
  outputModalities: [],
  providers: [],
  vendors: [],
  capabilities: [],
  minContext: 0,
  maxInput: 0,
  pricing: "all",
  source: "all",
  releasedWithinDays: 0,
}

/** How many narrowing choices are in force, for the rail's toggle to say so. */
export function activeFilterCount(filters: CatalogFilters): number {
  return (
    filters.inputModalities.length +
    filters.outputModalities.length +
    filters.providers.length +
    filters.vendors.length +
    filters.capabilities.length +
    (filters.minContext > 0 ? 1 : 0) +
    (filters.maxInput > 0 ? 1 : 0) +
    (filters.pricing !== "all" ? 1 : 0) +
    (filters.source !== "all" ? 1 : 0) +
    (filters.releasedWithinDays > 0 ? 1 : 0)
  )
}

function pricingMatches(model: CatalogModelSummary, pricing: string): boolean {
  switch (pricing) {
    case "custom":
      return model.price_sources.some(
        (source) => source === "deployment" || source === "organization",
      )
    case "default":
      return model.price_sources.includes("defaults")
    case "priced":
      return model.min_input_price_per_million != null
    case "unpriced":
      return model.unpriced_count > 0
    default:
      return true
  }
}

function matchesQuery(model: CatalogModelSummary, query: string): boolean {
  // The selectors and instances are searched too: an operator's query is as
  // often `accounts/fireworks/models/glm-5p3` or `nebius` as it is a name.
  return (
    model.name.toLowerCase().includes(query) ||
    (model.vendor ?? "").toLowerCase().includes(query) ||
    model.id.includes(query) ||
    model.selectors.some((selector) =>
      selector.toLowerCase().includes(query),
    ) ||
    model.providers.some((provider) => provider.toLowerCase().includes(query))
  )
}

export function filterModels(
  models: CatalogModelSummary[],
  filters: CatalogFilters,
  now: Date = new Date(),
): CatalogModelSummary[] {
  const query = filters.query.trim().toLowerCase()
  const capabilities = CAPABILITY_FILTERS.filter((entry) =>
    filters.capabilities.includes(entry.value),
  )
  const releasedAfter =
    filters.releasedWithinDays > 0
      ? new Date(now.getTime() - filters.releasedWithinDays * DAY_MS)
          .toISOString()
          .slice(0, 10)
      : null
  return models.filter((model) => {
    if (query && !matchesQuery(model, query)) return false
    if (
      filters.inputModalities.length > 0 &&
      !filters.inputModalities.every((modality) =>
        model.input_modalities.includes(modality),
      )
    ) {
      return false
    }
    if (
      filters.outputModalities.length > 0 &&
      !filters.outputModalities.every((modality) =>
        model.output_modalities.includes(modality),
      )
    ) {
      return false
    }
    if (
      filters.vendors.length > 0 &&
      !filters.vendors.includes(model.vendor ?? "")
    ) {
      return false
    }
    if (
      filters.providers.length > 0 &&
      !filters.providers.some((provider) => model.providers.includes(provider))
    ) {
      return false
    }
    if (!capabilities.every((capability) => capability.test(model))) {
      return false
    }
    if (
      filters.minContext > 0 &&
      (model.context_window == null ||
        model.context_window < filters.minContext)
    ) {
      return false
    }
    if (!pricingMatches(model, filters.pricing)) return false
    if (filters.source === "discovered" && !model.discovered) return false
    if (filters.source === "custom" && model.discovered) return false
    if (
      filters.maxInput > 0 &&
      (model.min_input_price_per_million == null ||
        model.min_input_price_per_million > filters.maxInput)
    ) {
      return false
    }
    if (
      releasedAfter !== null &&
      (model.release_date == null || model.release_date < releasedAfter)
    ) {
      return false
    }
    return true
  })
}

export type CatalogSortColumn =
  | "name"
  | "released"
  | "input"
  | "output"
  | "context"
  | "providers"

// Unpriced or undated rows sort last whichever way the column goes, and ties
// fall back to the name so the order never depends on how the rows arrived.
export function compareModels(
  column: CatalogSortColumn,
  direction: "asc" | "desc",
): (a: CatalogModelSummary, b: CatalogModelSummary) => number {
  const sign = direction === "asc" ? 1 : -1
  const byName = (a: CatalogModelSummary, b: CatalogModelSummary) =>
    a.name.localeCompare(b.name)
  if (column === "name") {
    return (a, b) => byName(a, b) * sign
  }
  const pick = (model: CatalogModelSummary): number | string | null => {
    if (column === "released") return model.release_date ?? null
    if (column === "input") return model.min_input_price_per_million ?? null
    if (column === "output") return model.min_output_price_per_million ?? null
    if (column === "context") return model.context_window ?? null
    return model.provider_count
  }
  return (a, b) => {
    const av = pick(a)
    const bv = pick(b)
    if (av == null && bv == null) return byName(a, b)
    if (av == null) return 1
    if (bv == null) return -1
    const order = av < bv ? -1 : av > bv ? 1 : 0
    return order * sign || byName(a, b)
  }
}

/** The sort menu's choices, each a column and a direction. */
export const SORT_OPTIONS: {
  value: string
  label: string
  column: CatalogSortColumn
  direction: "asc" | "desc"
}[] = [
  { value: "newest", label: "Newest", column: "released", direction: "desc" },
  { value: "name", label: "Name", column: "name", direction: "asc" },
  {
    value: "price-asc",
    label: "Price: low to high",
    column: "input",
    direction: "asc",
  },
  {
    value: "price-desc",
    label: "Price: high to low",
    column: "input",
    direction: "desc",
  },
  {
    value: "context",
    label: "Context: high to low",
    column: "context",
    direction: "desc",
  },
  {
    value: "providers",
    label: "Most providers",
    column: "providers",
    direction: "desc",
  },
]

/** Vendors present, with the unknown bucket named, for the rail. */
export function vendorOptions(
  models: CatalogModelSummary[],
): { value: string; label: string }[] {
  const vendors = new Set(models.map((model) => model.vendor ?? ""))
  return [...vendors]
    .sort((a, b) => a.localeCompare(b))
    .map((vendor) => ({ value: vendor, label: vendor || "Unknown vendor" }))
}

/** Provider instances present, once each, for the rail. */
export function providerOptions(
  models: CatalogModelSummary[],
): { value: string; label: string }[] {
  const providers = new Set(models.flatMap((model) => model.providers))
  return [...providers]
    .sort((a, b) => a.localeCompare(b))
    .map((provider) => ({ value: provider, label: provider }))
}

export function priceSourceLabel(
  source: CatalogOffering["price_source"],
): string {
  switch (source) {
    case "organization":
      return "your rate"
    case "deployment":
      return "custom"
    case "defaults":
      return "default"
    default:
      return "not priced"
  }
}

export function credentialLabel(
  credential: CatalogOffering["credential"],
): string {
  switch (credential) {
    case "organization":
      return "your key"
    case "hosted":
      return "hosted"
    default:
      return "deployment"
  }
}

/**
 * The offering a snippet should name: the cheapest priced one, else the first.
 *
 * The detail lists offerings cheapest first already, so this is its first row;
 * spelled out so the "use this model" panel and the table cannot disagree.
 */
export function defaultOffering(
  offerings: CatalogOffering[],
): CatalogOffering | undefined {
  return offerings[0]
}

/**
 * The input and output rate a request of `atContext` tokens is metered at:
 * the highest tier at or below it, each rate falling back to the base where
 * the tier leaves one unset. The gateway applies the same rule to the list.
 */
export function ratesAtContext(
  pricing: ModelPricingInfo,
  atContext: number,
): { input: number; output: number } {
  const base = {
    input: pricing.input_price_per_million,
    output: pricing.output_price_per_million,
  }
  if (!atContext) return base
  const tier = (pricing.pricing_tiers ?? [])
    .filter(
      (
        candidate,
      ): candidate is { min_input_tokens: number } & Record<string, unknown> =>
        typeof candidate === "object" &&
        candidate !== null &&
        typeof (candidate as { min_input_tokens?: unknown })
          .min_input_tokens === "number" &&
        (candidate as { min_input_tokens: number }).min_input_tokens <=
          atContext,
    )
    .sort((a, b) => b.min_input_tokens - a.min_input_tokens)[0]
  if (!tier) return base
  const input = tier.input_price_per_million
  const output = tier.output_price_per_million
  return {
    input: typeof input === "number" ? input : base.input,
    output: typeof output === "number" ? output : base.output,
  }
}
