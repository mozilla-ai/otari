// What filled one model request's context window, as the blocks the context map
// draws. The provider's input-token count is exact; how it divides is either
// exact (the cache split every span carries) or an estimate (the character
// weights the gateway kept for a request it served, scaled to that count).
// Only numbers are ever involved: no part of a prompt reaches this module.

import { formatNumber, formatTokens } from "@/shared/helpers/format"

import { plural } from "./format"
import type { Observation } from "./types"

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
  /** The request's input tokens, all of it (output tokens are not part of the window it measures against). */
  used: number
  windowTokens?: number
  /** The split is an estimate (the total never is). */
  isEstimate: boolean
  /** The provider's exact cache counts, kept beside an estimated split rather than drawn in it. */
  cache?: CacheCounts
  /** The input is larger than the window it was measured against. */
  isOverWindow: boolean
}

/** Where the input came from, in the provider's own counts. */
export interface CacheCounts {
  read: number
  write: number
  fresh: number
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

/** A block's share of the window, one decimal, for the tooltip. */
export function formatBlockShare(share: number): string {
  return `${(share * 100).toFixed(1)}%`
}

/** The header's percentage: a whole number, `<1%` for a little, `>100%` past the window. */
export function formatUsedPercent(used: number, windowTokens: number): string {
  const percent = (used / windowTokens) * 100
  if (percent > 100) return ">100%"
  if (percent > 0 && percent < 0.5) return "<1%"
  return `${Math.round(percent)}%`
}

export const ESTIMATED = "Estimated from request size"

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
    system: ["System prompt", "System", []],
    tools: [
      "Tool definitions",
      "Tools",
      [plural(weights.toolCount, "tool definition")],
    ],
    files: [
      "Files",
      "Files",
      ["Images and documents, at a flat allowance each"],
    ],
    messages: [
      "Messages",
      "Messages",
      [plural(weights.messageCount, "message")],
    ],
    toolResults: ["Tool results", "Tool results", []],
  }
  return parts.flatMap((part, index) => {
    if (tokens[index] <= 0) return []
    const [title, category, notes] = labels[part.kind]
    return [{ kind: part.kind, title, category, tokens: tokens[index], notes }]
  })
}

function cacheCounts(input: ContextMapInput, inputTokens: number): CacheCounts {
  const read = Math.min(Math.max(input.cacheReadTokens ?? 0, 0), inputTokens)
  const write = Math.min(
    Math.max(input.cacheWriteTokens ?? 0, 0),
    inputTokens - read,
  )
  return { read, write, fresh: inputTokens - read - write }
}

function cacheSplit(input: ContextMapInput, inputTokens: number): Draft[] {
  const { read, write, fresh } = cacheCounts(input, inputTokens)
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

// The exact cache split of a request whose tiles are an estimate, when it used the
// cache at all. Counts that overrun the input cannot form a split, and clamping them
// would present a number the provider never reported as exact, so they are left out.
function exactCache(
  input: ContextMapInput,
  used: number,
): CacheCounts | undefined {
  const read = Math.max(input.cacheReadTokens ?? 0, 0)
  const write = Math.max(input.cacheWriteTokens ?? 0, 0)
  if (read + write > used) return undefined
  return read > 0 || write > 0
    ? { read, write, fresh: used - read - write }
    : undefined
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
    cache: fromWeights !== undefined ? exactCache(input, used) : undefined,
    isOverWindow,
  }
}

/** One model request's context window, from the token counts and weights its span kept. */
export function requestContextMap(
  request: Observation,
): ContextMap | undefined {
  return buildContextMap({
    inputTokens: request.promptTokens,
    cacheReadTokens: request.cachedTokens,
    cacheWriteTokens: request.cacheWriteTokens,
    windowTokens: request.contextWindow,
    weights: request.context,
  })
}

/** The requests a session's context is read from: the agent's last to record any input, and its fullest. */
export interface SessionRequests {
  latest: Observation
  fullest: Observation
  /** How many of the agent's own requests recorded any input. */
  count: number
}

const inputOf = (request: Observation) => request.promptTokens ?? 0

/**
 * Only the agent's own conversation carries the session's context. A subagent's
 * requests hold its own, and a side call to another model (a title, a quota
 * check) holds a few hundred tokens, so the agent's model is the one most of
 * its requests ran on.
 */
export function sessionRequests(
  observations: Observation[],
  inSubagent: (id: string) => boolean = () => false,
): SessionRequests | undefined {
  const candidates = observations.filter(
    (observation) =>
      observation.type === "GENERATION" &&
      inputOf(observation) > 0 &&
      !inSubagent(observation.id),
  )
  const runs = new Map<string | undefined, number>()
  for (const request of candidates)
    runs.set(request.model, (runs.get(request.model) ?? 0) + 1)
  let model: string | undefined
  let most = 0
  for (const [name, count] of runs)
    if (count > most) [model, most] = [name, count]
  const requests = candidates.filter((request) => request.model === model)
  const [first, ...rest] = requests
  if (!first) return undefined
  let latest = first
  let fullest = first
  for (const request of rest) {
    if (request.startMs >= latest.startMs) latest = request
    if (inputOf(request) > inputOf(fullest)) fullest = request
  }
  return { latest, fullest, count: requests.length }
}

/** The band's figure: what the request held, against its window where that is known. */
export function contextTotal(map: ContextMap): string {
  const used = formatBlockTokens(map.used)
  return map.windowTokens === undefined
    ? used
    : `${used} / ${formatWindow(map.windowTokens)}`
}

/** What the band counts across from its figure: how full the window is, and how full it got. */
export function contextCounts(
  map: ContextMap,
  { latest, fullest }: SessionRequests,
): string[] {
  // The note under the map says the window is unknown, and why that hides free space.
  if (map.windowTokens === undefined) return []
  if (map.isOverWindow) return ["Over the window"]
  const full = `${formatUsedPercent(map.used, map.windowTokens)} full`
  if (
    fullest === latest ||
    fullest.promptTokens === undefined ||
    !fullest.contextWindow
  )
    return [full]
  return [
    full,
    `fullest ${formatUsedPercent(fullest.promptTokens, fullest.contextWindow)}`,
  ]
}

// A session's context is its last request's, since each request carries the
// whole conversation so far. The fullest is named too, so a compaction late in
// the session does not hide how close it came to the window.
export function sessionNote({
  latest,
  fullest,
  count,
}: SessionRequests): string {
  const from =
    count > 1
      ? `From the last of ${formatNumber(count)} requests.`
      : "From the session's only request."
  if (fullest === latest || fullest.promptTokens === undefined) return from
  const share = fullest.contextWindow
    ? `, ${formatUsedPercent(fullest.promptTokens, fullest.contextWindow)} of its window`
    : ""
  return `${from} The fullest held ${formatBlockTokens(fullest.promptTokens)}${share}.`
}

/** What the map itself needs said: how exact its split is, what the cache held, and why free space may be missing. */
export function mapNote(map: ContextMap): string {
  const sentences = [
    map.isEstimate
      ? "The split is estimated from its size; the total is exact."
      : "The split is the provider's own. One by prompt, tools and messages is kept only for requests Otari served.",
  ]
  if (map.isEstimate && map.cache && map.cache.read > 0)
    sentences.push(
      `${formatUsedPercent(map.cache.read, map.used)} of it was read from the cache.`,
    )
  if (map.windowTokens === undefined)
    sentences.push(
      "This model's window is unknown, so free space is not shown.",
    )
  return sentences.join(" ")
}

// Approximate width of one character of the block text, and the padding around it, in px.
const CHAR_WIDTH = 7
const TEXT_PADDING = 16
const MIN_LABEL_HEIGHT = 36
const MIN_COUNT_HEIGHT = 56
const MIN_CATEGORY_WIDTH = 150

/** What a block of this size can write on itself. */
export interface BlockLabel {
  /** The title, or the shorter category when the title does not fit, or nothing. */
  text?: string
  count: boolean
  /** The category appended to the count, on a block wide enough that has one to add. */
  category: boolean
}

function fits(text: string, width: number): boolean {
  return text.length * CHAR_WIDTH + TEXT_PADDING <= width
}

/** Fits the longest wording of a block's name that its box can hold, in px. */
export function fitBlockLabel(
  block: Pick<ContextBlock, "title" | "category" | "tokens">,
  width: number,
  height: number,
): BlockLabel {
  const none = { count: false, category: false }
  if (height < MIN_LABEL_HEIGHT) return none
  const text = [block.title, block.category].find((name) => fits(name, width))
  if (text === undefined) return none
  const count =
    height >= MIN_COUNT_HEIGHT && fits(formatBlockTokens(block.tokens), width)
  return {
    text,
    count,
    category:
      count &&
      text === block.title &&
      block.category !== block.title &&
      width >= MIN_CATEGORY_WIDTH,
  }
}

/** The curve a block moves on: it is on screen already, so it eases out of one place and into the next. */
const MOVE_EASE = "cubic-bezier(0.77, 0, 0.175, 1)"
// The repo's `ease-out-fluid` token (cubic-bezier(0.32, 0.72, 0, 1)), for what enters.
const ENTER_EASE = "cubic-bezier(0.32, 0.72, 0, 1)"
const STAGGER_MS = 30
/** The most blocks that wait their turn to enter, so the last one settles within 300ms. */
export const MAX_STAGGER = 4

interface Motion {
  property: string
  duration: number
  ease: string
  delay: number
}

/**
 * The transition of one block, a row per property so a duration cannot drift
 * from the property it belongs to. Geometry is listed only while the data is
 * changing: while the container is resized the blocks follow it at once.
 */
export function blockTransition(
  index: number,
  animateGeometry: boolean,
): {
  transitionProperty: string
  transitionDuration: string
  transitionTimingFunction: string
  transitionDelay: string
} {
  const move = { duration: 200, ease: MOVE_EASE, delay: 0 }
  const rows: Motion[] = [
    ...(animateGeometry
      ? (["left", "top", "width", "height"] as const).map((property) => ({
          property,
          ...move,
        }))
      : []),
    { property: "transform", duration: 120, ease: ENTER_EASE, delay: 0 },
    { property: "border-color", duration: 120, ease: "ease", delay: 0 },
    {
      property: "opacity",
      duration: 180,
      ease: ENTER_EASE,
      delay: Math.min(index, MAX_STAGGER) * STAGGER_MS,
    },
  ]
  const join = (pick: (row: Motion) => string | number, unit = "") =>
    rows.map((row) => `${pick(row)}${unit}`).join(", ")
  return {
    transitionProperty: join((row) => row.property),
    transitionDuration: join((row) => row.duration, "ms"),
    transitionTimingFunction: join((row) => row.ease),
    transitionDelay: join((row) => row.delay, "ms"),
  }
}
