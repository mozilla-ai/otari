import type {
  CatalogCapabilities,
  CatalogModelSummary,
  CatalogOffering,
  CatalogQueryParams,
  CatalogVendorFacet,
} from "@/client"
import { providerDisplayName } from "@/shared/helpers/providers"

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

// The server evaluates these flags against each model, independently of its providers.
export const CAPABILITY_FILTERS: {
  value: string
  label: string
}[] = [
  {
    value: "tool_call",
    label: "Tool calling",
  },
  {
    value: "reasoning",
    label: "Reasoning",
  },
  {
    value: "structured_output",
    label: "Structured output",
  },
  {
    value: "attachment",
    label: "Attachments",
  },
  {
    value: "open_weights",
    label: "Open weights",
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

export type CatalogSortColumn =
  | "name"
  | "released"
  | "input"
  | "output"
  | "context"
  | "providers"

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

/**
 * The vendor slug a row's catalog id carries, or `undefined` when it carries
 * none.
 *
 * The gateway builds a model's id as `vendor_slug(vendor)/slug` and as the bare
 * slug where no vendor could be named (`model_identity.CatalogIdentity`), so the
 * prefix already is the normalized vendor and reading it off is what keeps a
 * second copy of those rules out of this codebase. Both halves are required:
 * `vendor` says a prefix was added, the separator says one is there to take.
 *
 * The prefix cannot contradict the vendor, which is what makes reading it safe:
 * `id` is a property computed from `vendor` on the one identity, and the
 * response sets both fields from that object. A row whose prefix named a
 * different company would not be a mark keyed wrong, it would be a response the
 * gateway cannot produce.
 */
export function makerKeyOf(
  model: Pick<CatalogModelSummary, "id" | "vendor">,
): string | undefined {
  if (!model.vendor) return undefined
  const separator = model.id.indexOf("/")
  return separator > 0 ? model.id.slice(0, separator) : undefined
}

/** The server's complete vendor choices, with their catalog identity slugs. */
export function vendorOptions(facets: CatalogVendorFacet[]) {
  return facets
    .map((facet) => ({
      value: facet.value,
      label: facet.value || "Unknown vendor",
      markKey: facet.vendor_slug ?? undefined,
    }))
    .sort((a, b) => a.label.localeCompare(b.label))
}

/** The server's complete provider choices, named for display. */
export function providerOptions(providers: string[]) {
  return providers
    .map((provider) => ({
      value: provider,
      label: providerDisplayName(provider),
    }))
    .sort((a, b) => a.label.localeCompare(b.label))
}

export function catalogRequest(
  filters: CatalogFilters,
  column: CatalogSortColumn,
  direction: "asc" | "desc",
  page: number,
  pageSize: number,
): CatalogQueryParams {
  return {
    skip: page * pageSize,
    limit: pageSize,
    search: filters.query.trim() || undefined,
    provider: [...filters.providers].sort(),
    vendor: [...filters.vendors].sort(),
    input_modality: [...filters.inputModalities].sort(),
    output_modality: [...filters.outputModalities].sort(),
    capability: [
      ...filters.capabilities,
    ].sort() as CatalogQueryParams["capability"],
    min_context: filters.minContext || undefined,
    max_input: filters.maxInput || undefined,
    pricing: filters.pricing as CatalogQueryParams["pricing"],
    source: filters.source as CatalogQueryParams["source"],
    released_within_days: filters.releasedWithinDays || undefined,
    sort: column,
    direction,
    include_facets: true,
  }
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
      return "org key"
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
