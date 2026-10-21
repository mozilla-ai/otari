/**
 * The rows a deployment tool pricing page prices, and the per-call rate rules,
 * kept apart from the rendering so they can be tested without mounting it.
 *
 * A gateway-run tool's call is priced at the first row that has a rate: a
 * search or fetch instance's own row (kind `search` or `fetch`, keyed by the
 * instance's name), then the tool's `otari:<tool>` row (kind `model`). Every
 * rate travels per million requests, so a dollar per call is stored as
 * 1,000,000; the page asks for dollars per call, as `docs/tools.md` says.
 */

import type { SearchToolsResponse } from "@/client"

export type ToolPricingService = "web_search" | "code_execution"

export type PriceKind = "model" | "search" | "fetch"

export interface ToolRateRow {
  kind: PriceKind
  /** The pricing row's `model_key`: `otari:<tool>`, or an instance's name. */
  modelKey: string
  /** What an operator calls it. */
  label: string
  /** The instance's provider, or `""` for an `otari:<tool>` row. */
  provider: string
  /** What the row prices, one line. */
  role: string
  /** An instance declared in config.yml, whose `price_per_call` rewrites its rate on every start. */
  isConfigured: boolean
}

/**
 * Whether the rate is the config file's to change: the instance is still
 * declared there and its rate in force was written from its `price_per_call`.
 * A rate saved here would be put back by the next start.
 */
export function isSetInConfig(
  row: ToolRateRow,
  origin: string | null | undefined,
): boolean {
  return row.isConfigured && origin === "config"
}

/** The `otari:<tool>` rows each page prices whatever the deployment configured. */
export const GATEWAY_ROWS: Readonly<Record<ToolPricingService, ToolRateRow[]>> =
  {
    web_search: [
      {
        kind: "model",
        modelKey: "otari:web_search",
        label: "Web search",
        provider: "",
        role: "Searches no instance prices, and searches on an organization's own search key.",
        isConfigured: false,
      },
      {
        kind: "model",
        modelKey: "otari:web_fetch",
        label: "Web fetch",
        provider: "",
        role: "Fetches no instance prices.",
        isConfigured: false,
      },
    ],
    code_execution: [
      {
        kind: "model",
        modelKey: "otari:code_execution",
        label: "Code execution",
        provider: "",
        role: "Each otari_code_execution call, on any sandbox.",
        isConfigured: false,
      },
    ],
  }

/**
 * One row per search or fetch instance a call can be served by.
 *
 * A config entry a stored instance shadows is left out: the stored one is in
 * effect, and both share the name the rate is keyed by.
 */
export function instanceRows(
  kind: "search" | "fetch",
  tools: SearchToolsResponse | undefined,
): ToolRateRow[] {
  if (!tools) return []
  const role =
    kind === "search"
      ? "Searches this instance serves, in the tool loop and on POST /search."
      : "Fetch tool calls this instance serves."
  const stored = tools.stored.map(
    (tool): ToolRateRow => ({
      kind,
      modelKey: tool.name,
      label: tool.name,
      provider: tool.provider,
      role,
      isConfigured: false,
    }),
  )
  const configured = tools.config
    .filter((tool) => !tool.shadowed)
    .map(
      (tool): ToolRateRow => ({
        kind,
        modelKey: tool.name,
        label: tool.name,
        provider: tool.provider,
        role,
        isConfigured: true,
      }),
    )
  return [...configured, ...stored]
}

export const rowId = (row: Pick<ToolRateRow, "kind" | "modelKey">) =>
  `${row.kind}:${row.modelKey}`

const PER_MILLION = 1_000_000

/** A stored per-million rate as dollars per call. */
export function perCall(stored: number): number {
  return stored / PER_MILLION
}

/**
 * A typed per-call price as the per-million value the API stores, or
 * `undefined` when it is not one.
 *
 * Digits with an optional decimal part, spelled out rather than left to
 * `Number`, which reads "1e3" as 1000 and "0x10" as 16. Rounded, because 0.07
 * per call is 70000.00000000001 per million in binary floating point.
 */
export function storedFromPerCall(text: string): number | undefined {
  const trimmed = text.trim()
  if (!/^\d+(\.\d+)?$/.test(trimmed)) return undefined
  return Math.round(Number(trimmed) * PER_MILLION)
}
