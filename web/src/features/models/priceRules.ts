import { formatPricedRate, pricingUnitOf } from "@/features/models/pricingUnit"

// A price's overrides and multipliers in words, for a reader. The gateway gives
// a row's `rates` block untyped, as it was written, so this reads it
// defensively: a part it cannot read is left out rather than guessed at.

/** One rule: what it changes, and the condition it changes it under. */
export interface PriceRule {
  change: string
  when: string
}

type Block = Record<string, unknown>

function isBlock(value: unknown): value is Block {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

function isNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value)
}

// How many of a unit one published price buys, as models.dev writes a price.
const PUBLISHED_PER: Readonly<Record<string, number>> = {
  token: 1_000_000,
  character: 1_000_000,
  web_search: 1_000,
}

function published(price: number, unit: string): string {
  const per = PUBLISHED_PER[unit] ?? 1
  return formatPricedRate((price * 1_000_000) / per, pricingUnitOf(unit))
}

/** What one override sets, each price named with the price it replaces. */
function overrideChange(override: Block, unit: string, path: string): string {
  const parts: string[] = []
  if (isNumber(override.price))
    parts.push(`${published(override.price, unit)} for ${path}`)
  if (isNumber(override.reasoning)) {
    const modality = path.split(" ")[0]
    parts.push(
      `${published(override.reasoning, unit)} for ${modality} reasoning`,
    )
  }
  const cache = isBlock(override.cache) ? override.cache : {}
  parts.push(...cacheChanges(cache, unit, `${path} cache`))
  if (isBlock(cache.explicit))
    parts.push(...cacheChanges(cache.explicit, unit, `${path} explicit cache`))
  return parts.join(", ")
}

/** What a `cache` block sets: reads, writes (one price, or one per TTL) and storage. */
function cacheChanges(cache: Block, unit: string, label: string): string[] {
  const parts: string[] = []
  if (isNumber(cache.read))
    parts.push(`${published(cache.read, unit)} for ${label} reads`)
  if (isNumber(cache.write))
    parts.push(`${published(cache.write, unit)} for ${label} writes`)
  if (isBlock(cache.write))
    for (const [ttl, price] of Object.entries(cache.write))
      if (isNumber(price))
        parts.push(`${published(price, unit)} for ${label} writes (${ttl})`)
  if (isNumber(cache.storage))
    parts.push(`${published(cache.storage, unit)} for ${label} storage`)
  return parts
}

const KEY_LABELS: Readonly<Record<string, string>> = {
  utc_days: "UTC day",
  utc_hours: "UTC time",
  input_video: "a reference video",
}

const KEY_SUFFIX: Readonly<Record<string, string>> = {
  context: " tokens",
  duration: " seconds",
}

const OPERATOR_WORDS: Readonly<Record<string, string>> = {
  equals: "is",
  gt: "is over",
  gte: "is at least",
  lt: "is under",
  lte: "is at most",
}

function operand(value: unknown, key: string): string {
  if (isNumber(value))
    return `${value.toLocaleString("en-US")}${KEY_SUFFIX[key] ?? ""}`
  return String(value)
}

/** One key's test, as "service tier is flex". */
function keyCondition(key: string, test: unknown): string {
  const label = KEY_LABELS[key] ?? key.replaceAll("_", " ")
  if (!isBlock(test)) return label
  const [operator, value] = Object.entries(test)[0] ?? []
  if (operator === "in" && Array.isArray(value))
    return `${label} is one of ${either(value.map((item) => operand(item, key)))}`
  if (operator === "min_items")
    return `${label} has at least ${operand(value, "")} items`
  const words = OPERATOR_WORDS[operator ?? ""]
  return words === undefined
    ? label
    : `${label} ${words} ${operand(value, key)}`
}

/**
 * A list read as alternatives: "a, b or c". No comma before the "or", and tests
 * are joined with a bare "and", so a list's commas never read as a test's.
 */
function either(items: readonly string[]): string {
  return items.length < 2
    ? (items[0] ?? "")
    : `${items.slice(0, -1).join(", ")} or ${items.at(-1)}`
}

/** Whether a condition joins more than one test, so it reads as a group inside another. */
function isGroup(when: unknown): boolean {
  if (!isBlock(when)) return false
  const entries = Object.entries(when)
  if (entries.length > 1) return true
  const [key, test] = entries[0] ?? []
  return (key === "all" || key === "any") && Array.isArray(test)
    ? test.length > 1
    : false
}

/** A condition as words: its keys and combinators, every one of which must hold. */
export function describeCondition(when: unknown): string {
  if (!isBlock(when)) return ""
  const entries = Object.entries(when)
  const parts = entries.map(([key, test]) => {
    if ((key === "all" || key === "any") && Array.isArray(test)) {
      const joined = test
        .map((inner) =>
          isGroup(inner)
            ? `(${describeCondition(inner)})`
            : describeCondition(inner),
        )
        .join(key === "all" ? " and " : " or ")
      // Beside other tests, a group keeps its bounds too.
      return entries.length > 1 && test.length > 1 ? `(${joined})` : joined
    }
    if (key === "not") return `not (${describeCondition(test)})`
    return keyCondition(key, test)
  })
  return parts.join(" and ")
}

/** The prices a factor scales, as "×2 on text output". */
function factorChange(factor: unknown, path: string[] = []): string[] {
  if (isNumber(factor))
    return [
      `×${factor} on ${path.length === 0 ? "every price" : path.join(" ")}`,
    ]
  if (!isBlock(factor)) return []
  return Object.entries(factor).flatMap(([step, inner]) =>
    factorChange(inner, [...path, step.replaceAll("_", " ")]),
  )
}

function overridesOf(price: unknown, path: string): PriceRule[] {
  if (!isBlock(price) || !Array.isArray(price.overrides)) return []
  const unit = typeof price.unit === "string" ? price.unit : "token"
  return price.overrides.flatMap((override) => {
    if (!isBlock(override)) return []
    const change = overrideChange(override, unit, path)
    return change === ""
      ? []
      : [{ change, when: describeCondition(override.when) }]
  })
}

/** Every override and multiplier `rates` holds, in the order the gateway applies them. */
export function priceRules(rates: unknown): PriceRule[] {
  if (!isBlock(rates)) return []
  const overrides = Object.entries(rates).flatMap(([key, prices]) => {
    if (key === "multipliers") return []
    if (key === "web_search")
      return overridesOf(
        { unit: "web_search", ...(isBlock(prices) ? prices : {}) },
        "web search",
      )
    if (!isBlock(prices)) return []
    return ["input", "output"].flatMap((direction) =>
      overridesOf(prices[direction], `${key} ${direction}`),
    )
  })
  const multipliers = Array.isArray(rates.multipliers)
    ? rates.multipliers.flatMap((multiplier) => {
        if (!isBlock(multiplier)) return []
        const change = factorChange(multiplier.factor).join(", ")
        return change === ""
          ? []
          : [{ change, when: describeCondition(multiplier.when) }]
      })
    : []
  return [...overrides, ...multipliers]
}
