import type { RadioOption } from "@/design-system/forms/RadioGroup"
import { formatRate, formatUnitRate } from "@/shared/helpers/format"

// What a stored rate is per. Every rate travels as `*_price_per_million`, so a
// per-request rate of $2 per 1,000 searches is stored as 2000. The forms here
// ask for a token or character rate per million and a request or image rate
// per thousand, the way providers publish them, and convert on the way in and
// out. The unit names are the API's, singular, as models.dev writes them.

export type PricingUnit = "token" | "request" | "image" | "character"

export const PRICING_UNIT_OPTIONS: readonly RadioOption[] = [
  { value: "token", label: "Tokens" },
  { value: "request", label: "Requests" },
  { value: "image", label: "Images" },
  { value: "character", label: "Characters" },
]

export const PRICING_UNIT_DESCRIPTION =
  "What the rate bills: tokens the provider reports, each call (or rerank search unit), each image returned, or each character of text sent to a speech model. Price transcription and moderation models per request."

// The plural names the API served before this release, read for one release
// in case a replica still running the old code answers.
const RETIRED_UNITS: Readonly<Record<string, PricingUnit>> = {
  tokens: "token",
  requests: "request",
  images: "image",
}

/** A stored unit as one this form knows, so an unknown value reads as tokens. */
export function pricingUnitOf(value: string | null | undefined): PricingUnit {
  const unit = RETIRED_UNITS[value ?? ""] ?? value
  return unit === "request" || unit === "image" || unit === "character"
    ? unit
    : "token"
}

// The unit a rate for a request on each pass-through endpoint is most likely in.
// A rate's unit decides what it bills: speech reports the characters it was
// sent, and transcription and moderations report no tokens, so a token rate
// there is read per request only until the gateway reads rates strictly. Every
// other endpoint is priced by its tokens.
const ENDPOINT_UNITS: Readonly<Record<string, PricingUnit>> = {
  "/v1/rerank": "request",
  "/v1/images/generations": "image",
  "/v1/images/edits": "image",
  "/v1/audio/transcriptions": "request",
  "/v1/audio/speech": "character",
  "/v1/moderations": "request",
}

/** The unit a price for a request on `endpoint` is most likely to be in. */
export function pricingUnitForEndpoint(
  endpoint: string | null | undefined,
): PricingUnit {
  return ENDPOINT_UNITS[endpoint ?? ""] ?? "token"
}

// How many units one entered rate covers: a million tokens or characters, or a
// thousand requests or images.
const ENTERED_PER: Readonly<Record<PricingUnit, number>> = {
  token: 1_000_000,
  request: 1_000,
  image: 1_000,
  character: 1_000_000,
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

/** The label of the one rate a per-request, per-image or per-character price carries. */
export function unitRateLabel(unit: PricingUnit): string {
  switch (unit) {
    case "image":
      return "Price per 1,000 images"
    case "character":
      return "Price per 1M characters"
    default:
      return "Price per 1,000 requests"
  }
}

/**
 * One stored per-million rate in the unit it is charged in: "$0.25 / 1M" for a
 * token rate, "$15.00 / 1M characters" for a character rate, "$0.002 per
 * request" for a request rate.
 */
export function formatPricedRate(stored: number, unit: PricingUnit): string {
  if (unit === "token") return `${formatRate(stored)} / 1M`
  if (unit === "character") return `${formatRate(stored)} / 1M characters`
  const each = formatUnitRate(stored / 1_000_000)
  return unit === "image" ? `${each} per image` : `${each} per request`
}
