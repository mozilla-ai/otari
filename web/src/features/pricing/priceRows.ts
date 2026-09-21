/**
 * The deployment price table's row shape and the two derivations over it.
 *
 * Apart from the rendering so they can be tested directly: the drift figure is
 * arithmetic an operator reads as a percentage and acts on, and the row merge is
 * where a stored rate meets the default it is measured against.
 */

import type { PricingDriftRow, PricingResponse } from "@/client"

export interface PriceRow {
  modelKey: string
  input: number
  output: number
  cacheRead: number | null
  tiers: number
  unit: string
  updatedAt: string
  /** Where this rate came from: `config`, `api`, or absent for a row older than the column. */
  origin: string | null
  /** How far the rate sits from today's genai-prices default, where one exists. */
  drift?: PricingDriftRow
}

/**
 * One row per priced model, from the price that is in force today.
 *
 * `/pricing/current` does the reduction the page used to do in the browser: the
 * history holds one row per `effective_at`, and only the newest one that has
 * taken effect is what a request is metered at. Reducing it here meant reading
 * every revision of every model to render a screenful.
 *
 * `drift` is a separate operator-only read, capped at its own 200 rows, so the
 * "vs default" column is populated for the models that read covers and blank
 * beyond them. That cap is older than this page's paging and unchanged by it.
 */
export function currentRows(
  page: readonly PricingResponse[],
  drift: readonly PricingDriftRow[] = [],
): PriceRow[] {
  const byKey = new Map(drift.map((row) => [row.model_key, row]))
  return page.map((live) => ({
    modelKey: live.model_key,
    input: live.input_price_per_million,
    output: live.output_price_per_million,
    cacheRead: live.cache_read_price_per_million,
    tiers: live.pricing_tiers.length,
    unit: live.unit,
    updatedAt: live.updated_at,
    origin: live.origin ?? null,
    drift: byKey.get(live.model_key),
  }))
}

/** A signed percentage, or the dash for a rate with nothing to compare to. */
export function formatDrift(delta: number | null | undefined): string {
  if (delta == null) return "—"
  const rounded = Math.round(delta)
  if (rounded === 0) return "±0%"
  return `${rounded > 0 ? "+" : "−"}${Math.abs(rounded)}%`
}

// Beyond this the stored rate is more than a rounding away from the default,
// and the cell says so in the danger ink.
export const DRIFT_NOTICE_PERCENT = 10
