/**
 * The price rules of the hosted-model form, kept apart from the rendering so
 * they can be tested without mounting the dialog.
 *
 * Prices are held as the strings the boxes contain, in the unit the operator
 * picked: USD per million tokens, or per thousand requests or images. Every
 * rate travels per million, so a request or image rate is converted on the way
 * in and out. Input and output come as a pair or not at all (a pricing row
 * stores them together); a request or image price is the input rate alone, sent
 * with a zero output and no cache rates. On a reprice, a cache box cleared to
 * blank clears the stored rate: the API reads an explicit null as "clear", and
 * the form was seeded with the current values.
 */

import type {
  CreateHostedModelRequest,
  HostedModel,
  UpdateHostedModelRequest,
} from "@/client"
import {
  type PricingUnit,
  pricingUnitOf,
  toEnteredRate,
  toStoredRate,
} from "@/features/models/pricingUnit"

export interface PriceDraft {
  model: string
  unit: PricingUnit
  input: string
  output: string
  cacheRead: string
  cacheWrite: string
}

export const EMPTY_DRAFT: PriceDraft = {
  model: "",
  unit: "tokens",
  input: "",
  output: "",
  cacheRead: "",
  cacheWrite: "",
}

const asText = (value: number | null | undefined) =>
  value === null || value === undefined ? "" : String(value)

/**
 * Seeded with what the model currently serves at, defaults included: editing
 * starts from the price in force, and saving it unchanged pins that rate as
 * the deployment's own.
 */
export function draftFrom(model: HostedModel): PriceDraft {
  const unit = pricingUnitOf(model.unit)
  const input = model.input_price_per_million
  return {
    model: model.model,
    unit,
    input: asText(
      input === null || input === undefined || unit === "tokens"
        ? input
        : toEnteredRate(input, unit),
    ),
    output: unit === "tokens" ? asText(model.output_price_per_million) : "",
    cacheRead:
      unit === "tokens" ? asText(model.cache_read_price_per_million) : "",
    cacheWrite:
      unit === "tokens" ? asText(model.cache_write_price_per_million) : "",
  }
}

/** Blank is a valid "no rate"; anything else must be a non-negative number. */
export function parsePrice(text: string): { ok: boolean; value?: number } {
  const raw = text.trim()
  if (raw === "") return { ok: true }
  const value = Number(raw)
  return Number.isFinite(value) && value >= 0
    ? { ok: true, value }
    : { ok: false }
}

export const PRICE_ERROR = "A non-negative number of US dollars."

export type PriceKey = keyof Omit<PriceDraft, "model" | "unit">

export interface DraftReading {
  parsed: Record<PriceKey, { ok: boolean; value?: number }>
  hasBasePair: boolean
  /** One of input/output without the other, which no pricing row can store. */
  halfAPair: boolean
  hasCacheRate: boolean
  allParse: boolean
}

export function readDraft(draft: PriceDraft): DraftReading {
  const parsed = {
    input: parsePrice(draft.input),
    output: parsePrice(draft.output),
    cacheRead: parsePrice(draft.cacheRead),
    cacheWrite: parsePrice(draft.cacheWrite),
  }
  // A request or image price is one rate: the hidden boxes are not read.
  if (draft.unit !== "tokens") {
    return {
      parsed,
      hasBasePair: parsed.input.value !== undefined,
      halfAPair: false,
      hasCacheRate: false,
      allParse: parsed.input.ok,
    }
  }
  return {
    parsed,
    hasBasePair:
      parsed.input.value !== undefined && parsed.output.value !== undefined,
    halfAPair:
      (parsed.input.value !== undefined) !==
      (parsed.output.value !== undefined),
    hasCacheRate:
      parsed.cacheRead.value !== undefined ||
      parsed.cacheWrite.value !== undefined,
    allParse: Object.values(parsed).every((price) => price.ok),
  }
}

export function canSubmitDraft(draft: PriceDraft, isEditing: boolean): boolean {
  const reading = readDraft(draft)
  if (!reading.allParse || reading.halfAPair) return false
  // A reprice is a price: the pair is what the row stores.
  if (isEditing) return reading.hasBasePair
  return (
    draft.model.trim() !== "" && (reading.hasBasePair || !reading.hasCacheRate)
  )
}

/** The rates a draft with its base pair writes, converted to per million. */
function priceFields(draft: PriceDraft) {
  const { parsed } = readDraft(draft)
  // The fallbacks are unreachable behind `canSubmitDraft`, which required the pair.
  const input = parsed.input.value ?? 0
  if (draft.unit !== "tokens") {
    return {
      input_price_per_million: toStoredRate(input, draft.unit),
      output_price_per_million: 0,
      cache_read_price_per_million: null,
      cache_write_price_per_million: null,
      unit: draft.unit,
    }
  }
  return {
    input_price_per_million: input,
    output_price_per_million: parsed.output.value ?? 0,
    cache_read_price_per_million: parsed.cacheRead.value ?? null,
    cache_write_price_per_million: parsed.cacheWrite.value ?? null,
    unit: draft.unit,
  }
}

/** The POST body: blanks are omitted, so an unpriced offering stays unpriced. */
export function createBody(draft: PriceDraft): CreateHostedModelRequest {
  if (!readDraft(draft).hasBasePair) return { model: draft.model }
  const {
    cache_read_price_per_million: cacheRead,
    cache_write_price_per_million: cacheWrite,
    ...base
  } = priceFields(draft)
  return {
    model: draft.model,
    ...base,
    ...(cacheRead !== null ? { cache_read_price_per_million: cacheRead } : {}),
    ...(cacheWrite !== null
      ? { cache_write_price_per_million: cacheWrite }
      : {}),
  }
}

/** The PATCH body: a blanked cache box clears the stored rate, explicitly. */
export function repriceBody(draft: PriceDraft): UpdateHostedModelRequest {
  return priceFields(draft)
}
