import type {
  CatalogCapabilities,
  CatalogModelSummary,
  CatalogOffering,
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
    value: "vision",
    label: "Vision",
    test: (m) => m.input_modalities.includes("image"),
  },
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
    value: "open_weights",
    label: "Open weights",
    test: (m) => m.open_weights,
  },
]

export const CONTEXT_OPTIONS = [
  { value: "0", label: "Any context" },
  { value: "32000", label: "≥ 32K" },
  { value: "128000", label: "≥ 128K" },
  { value: "200000", label: "≥ 200K" },
  { value: "1000000", label: "≥ 1M" },
]

export interface CatalogFilters {
  query: string
  vendor: string
  /** A provider instance, as the Providers page links here with; "all" for any. */
  provider: string
  capability: string
  minContext: number
}

export function filterModels(
  models: CatalogModelSummary[],
  filters: CatalogFilters,
): CatalogModelSummary[] {
  const query = filters.query.trim().toLowerCase()
  const capability = CAPABILITY_FILTERS.find(
    (entry) => entry.value === filters.capability,
  )
  return models.filter((model) => {
    if (
      query &&
      !model.name.toLowerCase().includes(query) &&
      !(model.vendor ?? "").toLowerCase().includes(query) &&
      !model.id.includes(query)
    ) {
      return false
    }
    if (filters.vendor !== "all" && (model.vendor ?? "") !== filters.vendor) {
      return false
    }
    if (
      filters.provider !== "all" &&
      !model.providers.includes(filters.provider)
    ) {
      return false
    }
    if (capability && !capability.test(model)) {
      return false
    }
    if (
      filters.minContext > 0 &&
      (model.context_window == null ||
        model.context_window < filters.minContext)
    ) {
      return false
    }
    return true
  })
}

export type CatalogSortColumn = "name" | "released" | "input" | "output"

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
    return model.min_output_price_per_million ?? null
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

/** The distinct vendors in a catalog, for the filter, unknown ones folded to "". */
export function vendorOptions(
  models: CatalogModelSummary[],
): { value: string; label: string }[] {
  const names = Array.from(
    new Set(models.map((model) => model.vendor ?? "")),
  ).sort((a, b) => a.localeCompare(b))
  return [
    { value: "all", label: "All vendors" },
    ...names.map((name) => ({
      value: name,
      label: name === "" ? "Unknown vendor" : name,
    })),
  ]
}

/** The distinct provider instances across a catalog, for the filter. */
export function providerOptions(
  models: CatalogModelSummary[],
): { value: string; label: string }[] {
  const names = Array.from(
    new Set(models.flatMap((model) => model.providers)),
  ).sort((a, b) => a.localeCompare(b))
  return [
    { value: "all", label: "All providers" },
    ...names.map((name) => ({ value: name, label: name })),
  ]
}

/** What an offering's price rung is called on screen. */
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

/** Whose key an offering runs on, as a word. */
export function credentialLabel(
  credential: CatalogOffering["credential"],
): string {
  return credential === "organization" ? "your key" : "deployment"
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
