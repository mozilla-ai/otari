/**
 * The rules the Web search page shows, kept apart from the rendering so they
 * can be tested without mounting it: which tools a table lists and in what
 * order, what a tool's price reads as on each door, what a new tool may be
 * named, and how a tool's options travel between typed fields and JSON.
 */

import type {
  SearchProviderInfo,
  SearchProviderOption,
  SearchToolsResponse,
  SearchToolTestResponse,
  StoredSearchTool,
} from "@/client"
import type { RateRead } from "@/features/toolPricing/toolRates"

export type WebToolKind = "search" | "fetch"

/** The one fetch tool that always exists: the built-in fetcher. */
export const BUILTIN_FETCH = "builtin_fetch"

/** Names no tool may take, matched in any case except `*`. */
const RESERVED_NAMES = ["builtin_fetch", "none"]
const ALL_TOOLS = "*"

/** One search or fetch tool, as a table row shows it. */
export interface WebToolRow {
  kind: WebToolKind
  name: string
  provider: string
  /** `stored` rows are edited here; `config` and `builtin` ones are not. */
  source: "stored" | "config" | "builtin"
  /** `""` when the tool calls its provider's own endpoint. */
  apiBase: string
  /** The stored key's last four, `""` when there is none or it is not shown. */
  last4: string
  hasKey: boolean
  /** False when the deployment cannot decrypt the stored key. */
  isKeyReadable: boolean
  /** A stored tool that replaces the config.yml entry of the same name. */
  overridesConfig: boolean
  /** A search tool's enrichment fetch tool, `""` for the fetch default. */
  fetchTool: string
  /** Seconds, `undefined` for the gateway's own. */
  timeout: number | undefined
  options: Record<string, unknown>
  /** The row as stored, for an edit's concurrency check. */
  stored: StoredSearchTool | undefined
}

/**
 * Every tool a call can be served by, in the order the table lists them.
 *
 * `builtin_fetch` first, then the config.yml tools, then the stored ones. A
 * config entry a stored tool shadows is left out: the stored one is in effect
 * under the same name.
 */
export function listWebTools(
  kind: WebToolKind,
  tools: SearchToolsResponse | undefined,
): WebToolRow[] {
  if (!tools) return []
  const configured = tools.config
    .filter((tool) => !tool.shadowed)
    .map(
      (tool): WebToolRow => ({
        kind,
        name: tool.name,
        provider: tool.provider,
        source: tool.name === BUILTIN_FETCH ? "builtin" : "config",
        apiBase: tool.api_base ?? "",
        last4: "",
        hasKey: tool.has_api_key,
        isKeyReadable: true,
        overridesConfig: false,
        fetchTool: tool.fetch_tool ?? "",
        timeout: undefined,
        options: {},
        stored: undefined,
      }),
    )
  const stored = tools.stored.map((tool) => storedToolRow(kind, tool))
  return [...configured, ...stored]
}

/** A stored tool as a row, from the list or from an update's answer. */
export function storedToolRow(
  kind: WebToolKind,
  tool: StoredSearchTool,
): WebToolRow {
  return {
    kind,
    name: tool.name,
    provider: tool.provider,
    source: "stored",
    apiBase: tool.api_base ?? "",
    last4: tool.last4 ?? "",
    hasKey: Boolean(tool.last4),
    isKeyReadable: tool.decryptable,
    overridesConfig: tool.shadows_config,
    fetchTool: tool.fetch_tool ?? "",
    timeout: tool.timeout ?? undefined,
    options: tool.options ?? {},
    stored: tool,
  }
}

/** Whether a search default turns the model's search off: `none`, in any case. */
export function isNoSearch(value: string): boolean {
  return value.trim().toLowerCase() === "none"
}

/** What a connection test runs: a query for a search tool, a page for a fetch tool. */
export const TEST_INPUT: Readonly<
  Record<WebToolKind, { query: string | null; url: string | null }>
> = {
  search: { query: "otari web search", url: null },
  fetch: { query: null, url: "https://example.com" },
}

/** What a connection test answered, in a line. */
export function testOutcome(
  kind: WebToolKind,
  result: SearchToolTestResponse,
): string {
  if (!result.ok) return `Failed: ${result.error ?? "unknown error"}.`
  if (kind === "search") {
    const hits = result.hits ?? 0
    return `Worked: ${hits} ${hits === 1 ? "hit" : "hits"}.`
  }
  return `Worked: ${result.characters ?? 0} characters of page text.`
}

/** A per-million rate as the dollars per call a price field holds, never in exponent form. */
export function perCallText(perMillion: number): string {
  return (perMillion / 1_000_000).toFixed(12).replace(/\.?0+$/, "")
}

/** The `otari:` price a tool with no price of its own falls back to, by kind. */
export const FALLBACK_KEY: Readonly<Record<WebToolKind, string>> = {
  search: "otari:web_search",
  fetch: "otari:web_fetch",
}

/** What a tool's calls are billed at, door by door, once both reads answered. */
export type ToolPrice =
  /**
   * Its own rate, on every door. `fromConfig` when the rate in force came from
   * `price_per_call` in config.yml, which the next start writes back.
   */
  | { state: "own"; perMillion: number; fromConfig: boolean }
  /**
   * No rate of its own, and the fallback prices it where one applies. A fetch
   * tool has no other door; a search tool's calls on the search API are
   * priced by its own rate alone, so they go unpriced (`direct: "unpriced"`).
   */
  | {
      state: "fallback"
      perMillion: number
      fallbackKey: string
      direct: "none" | "unpriced"
    }
  | { state: "unpriced" }
  | { state: "pending" }
  | { state: "error" }

/**
 * A rate with nothing to fall back to: an `otari:` row's own. Never marked as
 * config.yml's: a `pricing:` entry is written once, so the page may change it.
 */
export function rateOnly(read: RateRead | undefined): ToolPrice {
  if (!read || read.isPending) return { state: "pending" }
  if (read.isError) return { state: "error" }
  if (!read.data) return { state: "unpriced" }
  return {
    state: "own",
    perMillion: read.data.input_price_per_million,
    fromConfig: false,
  }
}

/**
 * A tool's price, from its own rate read and its fallback's. Only a config.yml
 * tool's rate can be its `price_per_call`, which every start writes back.
 */
export function priceOf(
  tool: Pick<WebToolRow, "kind" | "source">,
  own: RateRead | undefined,
  fallback: RateRead | undefined,
): ToolPrice {
  const { kind } = tool
  if (!own || own.isPending) return { state: "pending" }
  if (own.isError) return { state: "error" }
  if (own.data) {
    return {
      state: "own",
      perMillion: own.data.input_price_per_million,
      fromConfig: tool.source === "config" && own.data.origin === "config",
    }
  }
  // Not "unpriced" until the rate it would inherit is known to be absent too.
  if (!fallback || fallback.isPending) return { state: "pending" }
  if (fallback.isError) return { state: "error" }
  if (fallback.data) {
    return {
      state: "fallback",
      perMillion: fallback.data.input_price_per_million,
      fallbackKey: FALLBACK_KEY[kind],
      direct: kind === "search" ? "unpriced" : "none",
    }
  }
  return { state: "unpriced" }
}

/**
 * Whether some call on the tool goes unpriced, which `require_pricing` refuses:
 * no rate at all, or a search tool whose search API calls have no rate of
 * their own. Unknown while a read is pending or failed.
 */
export function hasUnpricedCalls(price: ToolPrice): boolean {
  return (
    price.state === "unpriced" ||
    (price.state === "fallback" && price.direct === "unpriced")
  )
}

/**
 * The name a new tool gets when its name is left blank: the provider's, or,
 * for a fetch tool whose provider also has a search tool by that name,
 * `<provider>-fetch` (names are unique across both kinds). A number follows
 * when that is taken too.
 */
export function suggestName(
  kind: WebToolKind,
  provider: string,
  taken: readonly string[],
): string {
  if (provider === "") return ""
  const used = new Set(taken)
  const base =
    kind === "fetch" && used.has(provider) ? `${provider}-fetch` : provider
  if (!used.has(base)) return base
  for (let n = 2; ; n++) {
    if (!used.has(`${base}-${n}`)) return `${base}-${n}`
  }
}

/** Why a tool may not take `name`, or `""` when it may. Mirrors the API's checks. */
export function nameProblem(name: string, taken: readonly string[]): string {
  if (name === "") return ""
  if (/[/:]/.test(name)) return "A name cannot contain ':' or '/'."
  if (name === ALL_TOOLS || RESERVED_NAMES.includes(name.toLowerCase())) {
    return "builtin_fetch, none and * are reserved."
  }
  if (taken.includes(name)) {
    return "Another search or fetch tool already has this name."
  }
  return ""
}

/** An option the form gives a field of its own; the rest go in the JSON box. */
export function isTypedOption(option: SearchProviderOption): boolean {
  return option.type !== "array" && option.type !== "object"
}

/** The options a provider's form shows as fields, operator-only ones included. */
export function typedOptions(
  provider: SearchProviderInfo | undefined,
): SearchProviderOption[] {
  return (provider?.options ?? []).filter(isTypedOption)
}

/** A typed option's draft as the text its field holds, `""` for unset. */
function optionText(value: unknown): string {
  if (value === undefined || value === null) return ""
  if (typeof value === "boolean") return value ? "true" : "false"
  return String(value)
}

/**
 * A tool's stored options split into the typed fields' drafts and the JSON
 * box's text, so an edit opens with every value where the form shows it.
 */
export function splitOptions(
  provider: SearchProviderInfo | undefined,
  options: Record<string, unknown>,
): { typed: Record<string, string>; extraJson: string } {
  const typed: Record<string, string> = {}
  const extra: Record<string, unknown> = {}
  const fields = new Set(typedOptions(provider).map((option) => option.name))
  for (const [key, value] of Object.entries(options)) {
    const isScalar = value === null || typeof value !== "object"
    if (fields.has(key) && isScalar) typed[key] = optionText(value)
    else extra[key] = value
  }
  return {
    typed,
    extraJson:
      Object.keys(extra).length > 0 ? JSON.stringify(extra, null, 2) : "",
  }
}

const INTEGER = /^-?\d+$/
const NUMBER = /^-?\d+(\.\d+)?$/

/** A typed field's draft as the value sent, or why it cannot be sent. */
function parseOption(
  option: SearchProviderOption,
  text: string,
): { value: unknown } | { error: string } {
  if (option.type === "boolean") return { value: text === "true" }
  if (option.type === "integer") {
    return INTEGER.test(text)
      ? { value: Number(text) }
      : { error: "A whole number." }
  }
  if (option.type === "number") {
    return NUMBER.test(text) ? { value: Number(text) } : { error: "A number." }
  }
  return { value: text }
}

/**
 * The options to send: the JSON box's object, then each filled field over it.
 * A field left blank is not sent, so the provider's own default applies.
 */
export function buildOptions(
  provider: SearchProviderInfo | undefined,
  typed: Record<string, string>,
  extraJson: string,
):
  | { ok: true; options: Record<string, unknown> }
  | { ok: false; fieldErrors: Record<string, string>; jsonError: string } {
  const fieldErrors: Record<string, string> = {}
  let jsonError = ""
  let options: Record<string, unknown> = {}
  if (extraJson.trim() !== "") {
    try {
      const parsed: unknown = JSON.parse(extraJson)
      if (
        parsed === null ||
        typeof parsed !== "object" ||
        Array.isArray(parsed)
      ) {
        jsonError = 'A JSON object, such as {"contents": {"text": true}}.'
      } else {
        options = { ...(parsed as Record<string, unknown>) }
      }
    } catch {
      jsonError = "Not valid JSON."
    }
  }
  for (const option of typedOptions(provider)) {
    const text = (typed[option.name] ?? "").trim()
    if (text === "") continue
    const parsed = parseOption(option, text)
    if ("error" in parsed) fieldErrors[option.name] = parsed.error
    else options[option.name] = parsed.value
  }
  if (jsonError !== "" || Object.keys(fieldErrors).length > 0) {
    return { ok: false, fieldErrors, jsonError }
  }
  return { ok: true, options }
}

/** The options a provider takes only as JSON, named in the box's help. */
export function jsonOnlyOptions(
  provider: SearchProviderInfo | undefined,
): string[] {
  return (provider?.options ?? [])
    .filter((option) => !isTypedOption(option))
    .map((option) => option.name)
}

/**
 * Whether the model can search with a tool on this provider. The catalog gives
 * a provider no tier when the search library has no adapter for it, such as
 * `searxng`: the search API serves it through its own client, while the
 * model's search needs the library.
 */
export function servesTheModel(provider: SearchProviderInfo | undefined) {
  return provider === undefined || Boolean(provider.tier)
}

/** A value with its object keys sorted and its unset (null) members dropped. */
function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`
  if (value !== null && typeof value === "object") {
    const members = Object.entries(value)
      .filter(([, member]) => member !== null && member !== undefined)
      .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
      .map(([key, member]) => `${JSON.stringify(key)}:${canonical(member)}`)
    return `{${members.join(",")}}`
  }
  return JSON.stringify(value)
}

/**
 * Whether two option maps say the same thing: key order and unset values do
 * not count, so an edit that leaves the options alone does not send them.
 */
export function sameOptions(
  a: Record<string, unknown>,
  b: Record<string, unknown>,
): boolean {
  return canonical(a) === canonical(b)
}

/** A short line for a tool's options, for its row's details. */
export function describeOptions(options: Record<string, unknown>): string {
  const entries = Object.entries(options)
  if (entries.length === 0) return "Provider defaults"
  return entries
    .map(([key, value]) =>
      typeof value === "object" && value !== null
        ? `${key} ${JSON.stringify(value)}`
        : `${key} ${String(value)}`,
    )
    .join(" · ")
}
