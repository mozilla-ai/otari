import { formatRate, formatUnitRate } from "@/shared/helpers/format"

import type { ModelType } from "./catalog"
import { type PricingUnit, pricesMadeMedia, pricingUnitOf } from "./pricingUnit"

/** A kind a provider's model list files a model under: `unknown` where no source names one. */
export type OfferedKind = ModelType | "unknown"

/** The kind a provider's model list is narrowed to; empty for every kind. */
export type KindFilter = OfferedKind | ""

/** Each kind's tab, in the order the tabs are shown. */
const KIND_LABELS: Readonly<Record<OfferedKind, string>> = {
  chat: "Text",
  image: "Image",
  video: "Video",
  speech: "Speech",
  transcription: "Transcription",
  decision: "Decisions",
  embedding: "Embeddings",
  rerank: "Rerank",
  moderation: "Moderation",
  realtime: "Realtime",
  other: "Other",
  unknown: "Unknown",
}

const OFFERED_KINDS = Object.keys(KIND_LABELS) as OfferedKind[]

/** A URL value read as a kind, and anything else as every kind. */
export function kindFilterOf(value: string): KindFilter {
  return (OFFERED_KINDS as readonly string[]).includes(value)
    ? (value as OfferedKind)
    : ""
}

export interface KindTab {
  kind: KindFilter
  label: string
  count: number
}

/**
 * The tabs over a provider's models: All, then each kind it offers, with its
 * count. The kind shown keeps its tab at a count of zero, so a refresh that
 * empties it does not leave the list under a tab that is gone.
 */
export function kindTabs(
  counts: Readonly<Record<string, number>> | undefined,
  shown: KindFilter,
): KindTab[] {
  const total = Object.values(counts ?? {}).reduce((sum, n) => sum + n, 0)
  const tabs: KindTab[] = [{ kind: "", label: "All", count: total }]
  for (const kind of OFFERED_KINDS) {
    const count = counts?.[kind] ?? 0
    if (count > 0 || kind === shown) {
      tabs.push({ kind, label: KIND_LABELS[kind], count })
    }
  }
  return tabs
}

/** The rates a provider's model list carries for one model. */
export interface OfferedRates {
  /** The model's kind, which says whether a per-second price is video or audio. */
  type?: string | null
  unit?: string | null
  input_price_per_million?: number | null
  output_price_per_million?: number | null
  cache_read_price_per_million?: number | null
  cache_write_price_per_million?: number | null
}

export type PriceSide = "input" | "output" | "cache_read" | "cache_write"

/**
 * One price cell: the amount, and for a price not per token, what it prices
 * (`title`), its unit (`per`) and the amount with its unit (`rate`).
 */
export interface OfferedPrice {
  amount: string
  title?: string
  per?: string
  rate?: string
}

// A price for media a model makes, else undefined. Stored in the input column,
// since it is the row's one price, and shown under Output, the side it bills.
// A second prices video only on a video model; on any other it is audio, which
// has an input and an output side as a minute does.
function madeTitle(unit: PricingUnit, type: string | null | undefined) {
  if (!pricesMadeMedia(unit, type)) return undefined
  return unit === "second" ? "Video output" : "Image output"
}

// The amount and its unit, from a rate stored per million of the unit.
function perUnit(stored: number, unit: PricingUnit): [string, string] {
  if (unit === "character") return [formatRate(stored), "/1M characters"]
  if (unit === "web_search") return [formatRate(stored / 1_000), "/1K searches"]
  return [formatUnitRate(stored / 1_000_000), `/${unit}`]
}

function sideTitle(unit: PricingUnit, side: "input" | "output"): string {
  if (unit === "minute" || unit === "second") return `Audio ${side}`
  if (unit === "character") return `Text ${side}`
  if (unit === "web_search") return "Web search"
  return "Request"
}

/** What a model's list row shows in one price column; undefined for a dash. */
export function offeredPrice(
  row: OfferedRates,
  side: PriceSide,
): OfferedPrice | undefined {
  const unit = pricingUnitOf(row.unit)
  if (unit === "token") {
    const value = row[`${side}_price_per_million`]
    return value === null || value === undefined
      ? undefined
      : { amount: formatRate(value) }
  }
  if (side === "cache_read" || side === "cache_write") return undefined
  const made = madeTitle(unit, row.type)
  const value = made
    ? side === "output"
      ? row.input_price_per_million
      : undefined
    : row[`${side}_price_per_million`]
  if (value === null || value === undefined) return undefined
  const [amount, per] = perUnit(value, unit)
  return {
    amount,
    title: made ?? sideTitle(unit, side),
    per,
    rate: amount + per,
  }
}
