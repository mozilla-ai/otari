import { plural } from "./format"
import type { SessionStats, ToolTally } from "./sessionStats"

/** Past this share of the session, idle time is drawn shorter so the work beside it stays readable. */
export const IDLE_SHORTEN_AT = 0.5
/** Below this share of the session, idle time is left out: a few milliseconds between spans is not a pause. */
export const IDLE_SHOWN_AT = 0.01
/** The share of the bar that shortened idle time is drawn at. */
export const IDLE_DRAWN_AT = 0.25
/**
 * Past this share of tool time spent inside model requests, tools get a lane of their own
 * under the model's: a gateway's built-in tool runs inside the request that called it, so
 * as a slice of the bar it would count the same seconds twice.
 */
export const OWN_LANE_AT = 0.5
/** How many tools an opened tools row lists before folding the rest into one row. */
export const TOOLS_SHOWN = 5
/** How far parts may add up past the time they cover before the panel says they overlapped. */
const OVERLAP_SLACK = 0.01

export type PartKey =
  | "model"
  | "tools"
  | "waiting"
  | "hooks"
  | "idle"
  | "cacheRead"
  | "cacheWrite"
  | "output"
  | "input"
  | `model:${string}`

export interface Part {
  key: PartKey
  label: string
  hint?: string
  value: number
  /** Of the whole: the session's length, or what was priced. */
  share: number
  /** Of the bar. Differs from `share` only when idle is drawn shorter or parts overlap. */
  drawn: number
}

export interface TimeBreakdown {
  /** Work first, the longest first, then idle. */
  parts: Part[]
  /** The tools part when it ran inside the model's requests, drawn under the model's segment. */
  lane: Part | null
  idleShortened: boolean
  /** The work adds up to more than the time it covers, beyond what the lane explains. */
  overlaps: boolean
}

export const sum = (values: number[]) =>
  values.reduce((total, v) => total + v, 0)

export function timeBreakdown(stats: SessionStats): TimeBreakdown | null {
  const { lengthMs, activeMs, time } = stats
  if (activeMs <= 0) return null
  const ownLane =
    time.toolsMs > 0 && stats.toolsInModelMs / time.toolsMs > OWN_LANE_AT
  const work = [
    { key: "model" as const, label: "Model requests", value: time.modelMs },
    { key: "tools" as const, label: "Running tools", value: time.toolsMs },
    {
      key: "waiting" as const,
      label: "Waiting on you",
      hint: "to approve a call",
      value: time.waitingMs,
    },
    { key: "hooks" as const, label: "Hooks", value: time.hooksMs },
  ]
    .filter((part) => part.value > 0)
    .sort((a, b) => b.value - a.value)
  const gapMs = Math.max(0, lengthMs - activeMs)
  const idleMs = gapMs / lengthMs < IDLE_SHOWN_AT ? 0 : gapMs
  const idleShare = idleMs / lengthMs
  const idleShortened = idleShare > IDLE_SHORTEN_AT
  const idleDrawn = idleShortened ? IDLE_DRAWN_AT : idleShare

  // The work shares what idle leaves of the bar in proportion to its times, so
  // parts that overlapped still fill it without running past its end.
  const onBar = (key: PartKey) => !(ownLane && key === "tools")
  const barredMs = sum(work.filter((p) => onBar(p.key)).map((p) => p.value))
  const parts: Part[] = work.map((part) => ({
    ...part,
    share: part.value / lengthMs,
    drawn: onBar(part.key) ? ((1 - idleDrawn) * part.value) / barredMs : 0,
  }))
  if (idleMs > 0) {
    parts.push({
      key: "idle",
      label: "Idle",
      hint: stats.turns > 1 ? "between turns" : "between steps",
      value: idleMs,
      share: idleShare,
      drawn: idleDrawn,
    })
  }

  // The lane runs under the model's segment, as a share of it.
  const model = parts.find((part) => part.key === "model")
  const tools = parts.find((part) => part.key === "tools")
  const lane =
    ownLane && model && tools
      ? { ...tools, drawn: Math.min(1, tools.value / model.value) }
      : null
  return {
    parts,
    lane,
    idleShortened,
    overlaps: barredMs > activeMs * (1 + OVERLAP_SLACK),
  }
}

const METERS = [
  { key: "cacheRead", label: "Reading the cache" },
  { key: "cacheWrite", label: "Writing the cache" },
  { key: "output", label: "Output" },
  { key: "input", label: "Fresh input" },
] as const

/**
 * What the money paid for: by meter when the requests were priced by the token, else by
 * model, from what the gateway recorded for each request. Empty when nothing was priced.
 */
export function costParts(stats: SessionStats): {
  by: "meter" | "model"
  parts: Part[]
} {
  const { cost, tokens } = stats
  const byMeter = cost
    ? shares(
        METERS.map(({ key, label }) => ({
          key,
          label,
          hint: plural(tokens[key], "token"),
          value: cost[key],
        })),
      )
    : []
  if (byMeter.length > 0) return { by: "meter", parts: byMeter }
  return {
    by: "model",
    parts: shares(
      stats.models.map(({ model, requests, cost }) => ({
        key: `model:${model}` as const,
        label: model,
        hint: plural(requests, "request"),
        value: cost ?? 0,
      })),
    ),
  }
}

function shares(parts: Array<Omit<Part, "share" | "drawn">>): Part[] {
  const priced = parts.filter((part) => part.value > 0)
  const total = sum(priced.map((part) => part.value))
  return priced
    .sort((a, b) => b.value - a.value)
    .map((part) => ({
      ...part,
      share: part.value / total,
      drawn: part.value / total,
    }))
}

export interface ToolRows {
  /** The slowest tools, the slowest first. */
  shown: ToolTally[]
  /** The tools folded into one row, as a single tally; null when every tool is shown. */
  rest: { tools: number; calls: number; runningMs: number } | null
  /** The tools' own times add up to more than the time tools ran, because calls ran side by side. */
  overlap: boolean
}

export function toolRows(tools: ToolTally[], toolsMs: number): ToolRows {
  const slowest = [...tools].sort(
    (a, b) => b.runningMs - a.runningMs || b.calls - a.calls,
  )
  // Folding one tool into "1 more tool" saves no room, so the cut is only made
  // when it hides two or more.
  const cut = slowest.length > TOOLS_SHOWN + 1 ? TOOLS_SHOWN : slowest.length
  const folded = slowest.slice(cut)
  return {
    shown: slowest.slice(0, cut),
    rest:
      folded.length > 0
        ? {
            tools: folded.length,
            calls: sum(folded.map((tool) => tool.calls)),
            runningMs: sum(folded.map((tool) => tool.runningMs)),
          }
        : null,
    overlap:
      sum(tools.map((tool) => tool.runningMs)) > toolsMs * (1 + OVERLAP_SLACK),
  }
}
