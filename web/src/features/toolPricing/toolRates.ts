/**
 * The rows a deployment tool pricing page prices, and the per-call rate rules,
 * kept apart from the rendering so they can be tested without mounting it.
 *
 * A gateway-run tool's call is priced at its `otari:<tool>` row (kind `model`),
 * and a search or fetch tool's at its own row first (kind `search` or `fetch`,
 * keyed by the tool's name), which the Web search page sets. Every rate travels
 * per million requests, so a dollar per call is stored as 1,000,000; the pages
 * ask for dollars per call, as `docs/tools.md` says.
 */

import type { PricingResponse } from "@/client"

export type ToolPricingService = "code_execution"

export type PriceKind = "model" | "search" | "fetch"

export interface ToolRateRow {
  kind: PriceKind
  /** The pricing row's `model_key`: `otari:<tool>`, or a search or fetch tool's name. */
  modelKey: string
  /** What an operator calls it. */
  label: string
  /** What the row prices, one line. */
  role: string
}

/** The `otari:<tool>` rows each page prices whatever the deployment configured. */
export const GATEWAY_ROWS: Readonly<Record<ToolPricingService, ToolRateRow[]>> =
  {
    code_execution: [
      {
        kind: "model",
        modelKey: "otari:code_execution",
        label: "Code execution",
        role: "Each otari_code_execution call, on any sandbox.",
      },
    ],
  }

export const toRowId = (row: Pick<ToolRateRow, "kind" | "modelKey">) =>
  `${row.kind}:${row.modelKey}`

const PER_MILLION = 1_000_000

/** A stored per-million rate as dollars per call. */
export function toPerCall(stored: number): number {
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
export function parsePerCallPrice(text: string): number | undefined {
  const trimmed = text.trim()
  if (!/^\d+(\.\d+)?$/.test(trimmed)) return undefined
  return Math.round(Number(trimmed) * PER_MILLION)
}

/** One rate read, as a query answers it: `null` where nothing prices the key. */
export interface RateRead {
  data?: PricingResponse | null
  isPending: boolean
  isError: boolean
}

export interface PricedRow {
  row: ToolRateRow
  price: RateRead
}

/** Each row beside its rate read. */
export function pairWithRates(
  rows: readonly ToolRateRow[],
  reads: readonly RateRead[],
): PricedRow[] {
  return rows.map((row, index) => ({ row, price: reads[index] }))
}

/** Whether a row whose rate has been read is billed at none. */
export function hasUnpricedRow(entries: readonly PricedRow[]): boolean {
  return entries.some(
    ({ price }) => !price.isPending && !price.isError && !price.data,
  )
}
