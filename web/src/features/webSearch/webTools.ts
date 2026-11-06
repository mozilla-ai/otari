/**
 * The rules the Web search page shows, kept apart from the rendering so they
 * can be tested without mounting it: which tools a table lists and in what
 * order, what a tool's price reads as, what a new tool may be
 * named, and how a tool's options travel between typed fields and JSON.
 */

import type {
  SearchProviderInfo,
  SearchProviderOption,
  SearchToolsResponse,
  SearchToolTestResponse,
  StoredSearchTool,
} from "@/client"
import { type RateRead, toPerCall } from "@/features/toolPricing/toolRates"

export type WebToolKind = "search" | "fetch"

/** Both kinds, in the order a page lists them. */
export const WEB_TOOL_KINDS: readonly WebToolKind[] = ["search", "fetch"]

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

/** A tool as the listing gives it, with what one of its calls costs. */
export type ListedWebTool = WebToolRow & { price: ToolPrice }

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
): ListedWebTool[] {
  if (!tools) return []
  const configured = tools.config
    .filter((tool) => !tool.shadowed)
    .map(
      (tool): ListedWebTool => ({
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
        price: listedPrice(tool),
      }),
    )
  const stored = tools.stored.map(
    (tool): ListedWebTool => ({
      ...storedToolRow(kind, tool),
      price: listedPrice(tool),
    }),
  )
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

/** The `otari:` price of each kind, which prices a call on an organization's own key. */
export const ORGANIZATION_KEY_RATE: Readonly<Record<WebToolKind, string>> = {
  search: "otari:web_search",
  fetch: "otari:web_fetch",
}

/** The organization keys' row, by kind; its label also names the rate in the price dialog. */
export const ORGANIZATION_KEY_ROW: Readonly<
  Record<WebToolKind, { label: string; role: string }>
> = {
  search: {
    label: "Search on organization keys",
    role: "A search on an organization's own key, unless the organization sets its own price, and a search on the legacy settings.",
  },
  fetch: {
    label: "Fetch on organization keys",
    role: "A fetch on an organization's own key, unless the organization sets its own price.",
  },
}

/** What a tool's calls are billed at, once its rate is known. */
export type ToolPrice =
  /**
   * The rate that prices its calls, in dollars per call. `fromConfig` when
   * `price_per_call` in config.yml wrote it, which the next start writes back;
   * `oldKey` when it is stored under the tool's old `<provider>:<name>` model
   * key, which prices it until its own row is set.
   */
  | { state: "own"; perCall: number; fromConfig: boolean; oldKey: boolean }
  | { state: "unpriced" }
  | { state: "pending" }
  | { state: "error" }

/** A listed tool's price, as the gateway resolves it for the deployment's own calls. */
export function listedPrice(
  tool: Pick<
    SearchToolsResponse["config"][number],
    "price_per_call" | "price_source"
  >,
): ToolPrice {
  if (tool.price_per_call === null) return { state: "unpriced" }
  return {
    state: "own",
    perCall: tool.price_per_call,
    fromConfig: tool.price_source === "config",
    oldKey: tool.price_source === "old_key",
  }
}

/**
 * An `otari:` row's rate read as a price. Never marked as config.yml's: a
 * `pricing:` entry is written once, so the page may change it.
 */
export function rateOnly(read: RateRead | undefined): ToolPrice {
  if (!read || read.isPending) return { state: "pending" }
  if (read.isError) return { state: "error" }
  if (!read.data) return { state: "unpriced" }
  return {
    state: "own",
    perCall: toPerCall(read.data.input_price_per_million),
    fromConfig: false,
    oldKey: false,
  }
}

/** The `<provider>:<name>` model key, which prices a tool that has no row of its own. */
export function oldPriceKey(
  tool: Pick<WebToolRow, "provider" | "name">,
): string {
  return `${tool.provider}:${tool.name}`
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
