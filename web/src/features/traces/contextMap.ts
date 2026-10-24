// What filled one model request's context window, as the blocks the context map
// draws. The provider's input-token count is exact; how it divides is either
// exact (the cache split every span carries) or an estimate (the character
// weights the gateway kept for a request it served, scaled to that count).
// Only numbers are ever involved: no part of a prompt reaches this module.

import { formatTokens } from "@/shared/helpers/format"

/** The span attributes a gateway-served request's character weights are stored under. */
const ATTRIBUTE_PREFIX = "otari.context."

/** Characters (a flat allowance for each attachment) per kind of input, and how many definitions and messages there were. */
export interface ContextWeights {
  system: number
  tools: number
  user: number
  assistant: number
  toolResults: number
  files: number
  toolCount: number
  messageCount: number
}

export type ContextKind =
  | "system"
  | "tools"
  | "files"
  | "messages"
  | "toolResults"
  | "cacheRead"
  | "cacheWrite"
  | "fresh"
  | "free"

export interface ContextBlock {
  kind: ContextKind
  /** Bold label of the block. */
  title: string
  /** What kind of thing it is, appended to the count on a block big enough. */
  category: string
  tokens: number
  /** Of the window, or of the input when the window is unknown or was exceeded. */
  share: number
  /** Extra lines of the tooltip. */
  notes: string[]
}

export interface ContextMap {
  blocks: ContextBlock[]
  /** The request's input tokens, all of it. */
  used: number
  windowTokens?: number
  /** The split is an estimate (the total never is). */
  isEstimate: boolean
  /** The input is larger than the window it was measured against. */
  isOverWindow: boolean
}

export interface ContextMapInput {
  inputTokens?: number
  cacheReadTokens?: number
  cacheWriteTokens?: number
  windowTokens?: number
  weights?: ContextWeights
}

const WEIGHT_KEYS: [keyof ContextWeights, string][] = [
  ["system", "system_chars"],
  ["tools", "tools_chars"],
  ["user", "user_chars"],
  ["assistant", "assistant_chars"],
  ["toolResults", "tool_result_chars"],
  ["files", "file_chars"],
  ["toolCount", "tool_count"],
  ["messageCount", "message_count"],
]

/** A span's stored weights, or undefined when it kept none (OTLP, or a span from before they were kept). */
export function contextWeightsOf(
  attributes: Record<string, string | number | boolean>,
): ContextWeights | undefined {
  const weights: ContextWeights = {
    system: 0,
    tools: 0,
    user: 0,
    assistant: 0,
    toolResults: 0,
    files: 0,
    toolCount: 0,
    messageCount: 0,
  }
  let found = false
  for (const [field, key] of WEIGHT_KEYS) {
    const value = attributes[ATTRIBUTE_PREFIX + key]
    if (typeof value === "number" && Number.isFinite(value) && value >= 0) {
      weights[field] = Math.floor(value)
      found = true
    }
  }
  return found ? weights : undefined
}

/**
 * Splits `total` in proportion to `weights` into whole numbers that add up to
 * exactly `total`: each share is rounded down and the tokens left over go, one
 * each, to the largest fractional remainders (the larger weight first on a tie).
 */
export function scaleToTotal(
  weights: readonly number[],
  total: number,
): number[] {
  const sum = weights.reduce((acc, weight) => acc + Math.max(0, weight), 0)
  if (sum <= 0 || total <= 0) return weights.map(() => 0)
  const exact = weights.map((weight) => (Math.max(0, weight) * total) / sum)
  const scaled = exact.map(Math.floor)
  let left = total - scaled.reduce((acc, value) => acc + value, 0)
  const order = exact
    .map((value, index) => ({ index, remainder: value - Math.floor(value) }))
    .sort(
      (a, b) =>
        b.remainder - a.remainder ||
        weights[b.index] - weights[a.index] ||
        a.index - b.index,
    )
  for (const { index } of order) {
    if (left <= 0) break
    scaled[index] += 1
    left -= 1
  }
  return scaled
}

/** `14.8k`, `1.2M`, or a plain integer under a thousand. */
export function formatBlockTokens(value: number): string {
  const rounded = Math.round(value)
  // Rounding a count just under a million to one decimal must not read "1000.0k".
  if (rounded >= 999_950)
    return formatTokens(rounded >= 1_000_000 ? rounded : 1_000_000)
  return formatTokens(rounded)
}

/** A window as the header names it: `200k`, `1M`, no trailing `.0`. */
export function formatWindow(value: number): string {
  return formatBlockTokens(value).replace(".0", "")
}

/** A share of the window, one decimal, for the tooltip. */
export function formatShare(share: number): string {
  return `${(share * 100).toFixed(1)}%`
}

/** The header's percentage: a whole number, `<1%` for a little, `>100%` past the window. */
export function formatUsedPercent(used: number, windowTokens: number): string {
  const percent = (used / windowTokens) * 100
  if (percent > 100) return ">100%"
  if (percent > 0 && percent < 0.5) return "<1%"
  return `${Math.round(percent)}%`
}

const ESTIMATED = "Estimated from request size"

function plural(count: number, one: string, many: string): string {
  return `${count.toLocaleString("en-US")} ${count === 1 ? one : many}`
}

type Draft = Omit<ContextBlock, "share">

function composition(
  weights: ContextWeights,
  inputTokens: number,
): Draft[] | undefined {
  const parts = [
    { kind: "system" as const, chars: weights.system },
    { kind: "tools" as const, chars: weights.tools },
    { kind: "files" as const, chars: weights.files },
    { kind: "messages" as const, chars: weights.user + weights.assistant },
    { kind: "toolResults" as const, chars: weights.toolResults },
  ]
  if (parts.every((part) => part.chars === 0)) return undefined
  const tokens = scaleToTotal(
    parts.map((part) => part.chars),
    inputTokens,
  )
  const labels: Record<
    (typeof parts)[number]["kind"],
    [string, string, string[]]
  > = {
    system: ["System prompt", "System", [ESTIMATED]],
    tools: [
      "Tool definitions",
      "Tools",
      [
        plural(weights.toolCount, "tool definition", "tool definitions"),
        ESTIMATED,
      ],
    ],
    files: [
      "Files",
      "Files",
      ["Images and documents, at a flat allowance each", ESTIMATED],
    ],
    messages: [
      "Messages",
      "Messages",
      [plural(weights.messageCount, "message", "messages"), ESTIMATED],
    ],
    toolResults: ["Tool results", "Tool results", [ESTIMATED]],
  }
  return parts.flatMap((part, index) => {
    if (tokens[index] <= 0) return []
    const [title, category, notes] = labels[part.kind]
    return [{ kind: part.kind, title, category, tokens: tokens[index], notes }]
  })
}

function cacheSplit(input: ContextMapInput, inputTokens: number): Draft[] {
  const read = Math.min(Math.max(input.cacheReadTokens ?? 0, 0), inputTokens)
  const write = Math.min(
    Math.max(input.cacheWriteTokens ?? 0, 0),
    inputTokens - read,
  )
  const fresh = inputTokens - read - write
  const known =
    input.cacheReadTokens !== undefined || input.cacheWriteTokens !== undefined
  const drafts: Draft[] = []
  if (read > 0)
    drafts.push({
      kind: "cacheRead",
      title: "Cache read",
      category: "Cached",
      tokens: read,
      notes: ["Read from the provider's cache"],
    })
  if (write > 0)
    drafts.push({
      kind: "cacheWrite",
      title: "Cache write",
      category: "Cached",
      tokens: write,
      notes: ["Written to the provider's cache"],
    })
  if (fresh > 0)
    drafts.push({
      kind: "fresh",
      title: known ? "Fresh input" : "Input",
      category: "Input",
      tokens: fresh,
      notes: [
        known ? "Not served from the cache" : "Input tokens of the request",
      ],
    })
  return drafts
}

/**
 * The map of one request, or undefined when it has no input tokens to draw.
 * The blocks always add up to the input tokens exactly; the window, when known,
 * adds a Free block for what is left of it.
 */
export function buildContextMap(
  input: ContextMapInput,
): ContextMap | undefined {
  const used = input.inputTokens
  if (used === undefined || !Number.isFinite(used) || used <= 0)
    return undefined
  const windowTokens =
    input.windowTokens !== undefined && input.windowTokens > 0
      ? input.windowTokens
      : undefined
  const fromWeights = input.weights
    ? composition(input.weights, used)
    : undefined
  const drafts = fromWeights ?? cacheSplit(input, used)
  const isOverWindow = windowTokens !== undefined && used > windowTokens
  // A window the input overran says nothing of what is free, so shares are of the input.
  const whole =
    windowTokens !== undefined && !isOverWindow ? windowTokens : used
  const blocks: ContextBlock[] = drafts.map((draft) => ({
    ...draft,
    share: draft.tokens / whole,
  }))
  if (windowTokens !== undefined && !isOverWindow && windowTokens > used) {
    blocks.push({
      kind: "free",
      title: "Free",
      category: "Free",
      tokens: windowTokens - used,
      share: (windowTokens - used) / windowTokens,
      notes: ["The unused part of the context window"],
    })
  }
  return {
    blocks,
    used,
    windowTokens,
    isEstimate: fromWeights !== undefined,
    isOverWindow,
  }
}
