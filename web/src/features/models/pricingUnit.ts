import type { RadioOption } from "@/design-system/forms/RadioGroup"
import { formatRate, formatUnitRate } from "@/shared/helpers/format"

// What a stored rate is per. Every rate travels as `*_price_per_million`, so a
// per-request rate of $2 per 1,000 searches is stored as 2000. The forms here
// ask for a token rate per million and a request or image rate per thousand,
// the way providers publish them, and convert on the way in and out.

export type PricingUnit = "tokens" | "requests" | "images"

export const PRICING_UNIT_OPTIONS: readonly RadioOption[] = [
  { value: "tokens", label: "Tokens" },
  { value: "requests", label: "Requests" },
  { value: "images", label: "Images" },
]

/** A stored unit as one this form knows, so an unknown value reads as tokens. */
export function pricingUnitOf(value: string | null | undefined): PricingUnit {
  return value === "requests" || value === "images" ? value : "tokens"
}

// The one usage label a per-request price answers to without asking. Only the
// endpoints whose providers bill per call or per image are mapped, and every
// other endpoint is priced by its tokens.
const ENDPOINT_UNITS: Readonly<Record<string, PricingUnit>> = {
  "/v1/rerank": "requests",
  "/v1/images/generations": "images",
}

/** The unit a price for a request on `endpoint` is most likely to be in. */
export function pricingUnitForEndpoint(
  endpoint: string | null | undefined,
): PricingUnit {
  return ENDPOINT_UNITS[endpoint ?? ""] ?? "tokens"
}

// How many units one entered rate covers: a million tokens, or a thousand
// requests or images.
const ENTERED_PER: Readonly<Record<PricingUnit, number>> = {
  tokens: 1_000_000,
  requests: 1_000,
  images: 1_000,
}

// Rounded to the micro-dollar the gateway stores, so 0.07 per thousand is sent
// as 70 rather than 70.00000000000001.
function roundRate(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000
}

/** A rate the operator typed, as the per-million value the API stores. */
export function toStoredRate(entered: number, unit: PricingUnit): number {
  return roundRate((entered * 1_000_000) / ENTERED_PER[unit])
}

/** A stored per-million rate, as the value the form shows for `unit`. */
export function toEnteredRate(stored: number, unit: PricingUnit): number {
  return roundRate((stored * ENTERED_PER[unit]) / 1_000_000)
}

/** The label of the one rate a per-request or per-image price carries. */
export function unitRateLabel(unit: PricingUnit): string {
  return unit === "images"
    ? "Price per 1,000 images"
    : "Price per 1,000 requests"
}

/**
 * One stored per-million rate in the unit it is charged in: "$0.25 / 1M" for a
 * token rate, "$0.002 per request" for a request rate.
 */
export function formatPricedRate(stored: number, unit: PricingUnit): string {
  if (unit === "tokens") return `${formatRate(stored)} / 1M`
  const each = formatUnitRate(stored / 1_000_000)
  return unit === "images" ? `${each} per image` : `${each} per request`
}
