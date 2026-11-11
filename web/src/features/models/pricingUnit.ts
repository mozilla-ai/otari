import type { RatesDocument } from "@/client"
import type { RadioOption } from "@/design-system/forms/RadioGroup"
import { formatRate, formatUnitRate } from "@/shared/helpers/format"

// What a stored rate is per. Every rate travels as `*_price_per_million`, so a
// per-request rate of $2 per 1,000 searches is stored as 2000. The forms here
// ask for each rate the way providers publish it (a token or character rate
// per million, a request, image or search rate per thousand, a megapixel,
// second or minute rate per one) and convert on the way in and out. The unit
// names are the API's, singular, as models.dev writes them.

export type PricingUnit =
  | "token"
  | "request"
  | "image"
  | "character"
  | "megapixel"
  | "second"
  | "minute"
  | "web_search"

export const PRICING_UNIT_OPTIONS: readonly RadioOption[] = [
  { value: "token", label: "Tokens" },
  { value: "request", label: "Requests" },
  { value: "image", label: "Images" },
  { value: "megapixel", label: "Megapixels" },
  { value: "character", label: "Characters" },
  { value: "minute", label: "Minutes of audio" },
  { value: "second", label: "Seconds of video" },
  { value: "web_search", label: "Web searches" },
]

// A form that stores the per-million fields alone cannot price a second of
// video: the gateway stores a per-second price there on audio input (see
// `ratesForUnit`).
export const COLUMN_PRICING_UNIT_OPTIONS: readonly RadioOption[] =
  PRICING_UNIT_OPTIONS.filter((option) => option.value !== "second")

export const PRICING_UNIT_DESCRIPTION =
  "What the rate bills: tokens the provider reports, each call (or rerank search unit), each image or megapixel returned, each character of text sent to a speech model, each minute of audio transcribed, each second of video made, or each web search. Price moderation models per request."

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
  return PRICING_UNITS.has(unit ?? "") ? (unit as PricingUnit) : "token"
}

const PRICING_UNITS: ReadonlySet<string> = new Set(
  PRICING_UNIT_OPTIONS.map((option) => option.value),
)

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
  "/v1/videos": "second",
}

/** The unit a price for a request on `endpoint` is most likely to be in. */
export function pricingUnitForEndpoint(
  endpoint: string | null | undefined,
): PricingUnit {
  return ENDPOINT_UNITS[endpoint ?? ""] ?? "token"
}

// How many units one entered rate covers: a million tokens or characters, a
// thousand requests, images or searches, or one megapixel, second or minute.
const ENTERED_PER: Readonly<Record<PricingUnit, number>> = {
  token: 1_000_000,
  request: 1_000,
  image: 1_000,
  character: 1_000_000,
  megapixel: 1,
  second: 1,
  minute: 1,
  web_search: 1_000,
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

/** The label of the one rate a price in any unit but tokens carries. */
export function unitRateLabel(unit: PricingUnit): string {
  switch (unit) {
    case "image":
      return "Price per 1,000 images"
    case "character":
      return "Price per 1M characters"
    case "megapixel":
      return "Price per megapixel"
    case "second":
      return "Price per second of video"
    case "minute":
      return "Audio in, per minute"
    case "web_search":
      return "Price per 1,000 searches"
    default:
      return "Price per 1,000 requests"
  }
}

/**
 * Whether a price in `unit` has an output rate beside its input rate. A minute
 * of audio does: a model is priced on the audio it hears and the audio it
 * speaks. Every other unit but tokens is one rate.
 */
export function hasOutputRate(unit: PricingUnit): boolean {
  return unit === "minute"
}

/**
 * Whether a price is for media the model makes: an image, or a second of video
 * on a video model. Stored in the input column, as the row's one price, it
 * bills the output side. A second on any other kind of model is audio.
 */
export function pricesMadeMedia(
  unit: PricingUnit,
  type: string | null | undefined,
): boolean {
  return (
    unit === "image" ||
    unit === "megapixel" ||
    (unit === "second" && type === "video")
  )
}

/** The label of the output rate a unit with one carries. */
export function unitOutputRateLabel(unit: PricingUnit): string {
  return unit === "minute" ? "Audio out, per minute" : "Output"
}

// The word after "per" for a rate shown per one of its unit.
const EACH: Readonly<Partial<Record<PricingUnit, string>>> = {
  request: "request",
  image: "image",
  megapixel: "megapixel",
  second: "second",
  minute: "minute",
}

/**
 * One stored per-million rate in the unit it is charged in: "$0.25 / 1M" for a
 * token rate, "$15.00 / 1M characters" for a character rate, "$10.00 / 1K
 * searches" for a search rate, and "$0.002 per request" for a rate per one.
 */
export function formatPricedRate(stored: number, unit: PricingUnit): string {
  if (unit === "token") return `${formatRate(stored)} / 1M`
  if (unit === "character") return `${formatRate(stored)} / 1M characters`
  if (unit === "web_search")
    return `${formatRate(stored / 1_000)} / 1K searches`
  return `${formatUnitRate(stored / 1_000_000)} per ${EACH[unit] ?? "request"}`
}

/**
 * An offering's price in a line: "$0.50 in / $2.00 out" per million tokens,
 * or the one price of any other unit ("$0.04 per image"); "unpriced" where it
 * has none.
 */
export function formatOfferingPrice(
  pricing:
    | {
        input_price_per_million: number
        output_price_per_million: number
        unit?: string | null
      }
    | null
    | undefined,
): string {
  if (pricing == null) return "unpriced"
  const unit = pricingUnitOf(pricing.unit)
  if (unit !== "token")
    return formatPricedRate(pricing.input_price_per_million, unit)
  return `${formatRate(pricing.input_price_per_million)} in / ${formatRate(pricing.output_price_per_million)} out`
}

/**
 * The `rates` block to send for a price the per-million fields would store in
 * the wrong place, else nothing. The gateway stores a per-second column price
 * on audio input, which bills transcription; a video is billed from video
 * output, so a second of video is sent as a block.
 *
 * `stored` is the block of the row being edited, when it is in the same unit.
 * The field showed the price the columns are read from, the first slot priced
 * in the gateway's read order, so that slot is the one that moves. Audio is
 * sent as columns, which the gateway writes there. Any other slot moves in the
 * block, which keeps the row's other prices and its rules.
 */
export function ratesForUnit(
  entered: number,
  unit: PricingUnit,
  stored?: Readonly<Record<string, unknown>> | null,
): RatesDocument | undefined {
  if (unit !== "second") return undefined
  const price = { price: entered, unit: "second" }
  if (stored == null) return { video: { output: price } }
  const [modality, direction] = columnSlot(stored) ?? VIDEO_OUTPUT
  if (modality === "audio") return undefined
  const block = asBlock(stored[modality])
  return {
    ...stored,
    [modality]: {
      ...block,
      [direction]: { ...asBlock(block[direction]), ...price },
    },
  }
}

// The slots the gateway reads a row's columns from, first priced first, each
// with the direction the input column holds and then any other it reads.
const VIDEO_OUTPUT = ["video", "output"] as const
const COLUMN_READ_ORDER: readonly (readonly [string, string, ...string[]])[] = [
  ["text", "input", "output"],
  ["image", "output"],
  ["audio", "input", "output"],
  VIDEO_OUTPUT,
]

/**
 * Whether a stored block's columns are read from audio, which has an input and
 * an output rate per second as it has per minute.
 */
export function pricesAudio(
  stored: Readonly<Record<string, unknown>> | null | undefined,
): boolean {
  return stored != null && columnSlot(stored)?.[0] === "audio"
}

function columnSlot(stored: Readonly<Record<string, unknown>>) {
  return COLUMN_READ_ORDER.find(([name, ...directions]) =>
    directions.some((side) => isPriced(asBlock(stored[name])[side])),
  )
}

function isPriced(value: unknown): boolean {
  return asBlock(value).price != null
}

function asBlock(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {}
}
