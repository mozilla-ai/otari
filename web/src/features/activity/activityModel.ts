/**
 * What an activity row means, shared by the page and by the parts it renders.
 *
 * The page reads these to build its columns and its filters; the extracted
 * components read them to render one row, one cell or one detail panel. Split
 * out of `ActivityPage.tsx` so neither imports the other, and so the
 * derivations can be tested without rendering a page.
 *
 * The four formatters here (`formatUSD`, `formatTokenCount`,
 * `formatLatencyCell`, `formatElapsed`) are relocated unchanged and are
 * deliberately not reconciled with `shared/helpers/format.ts`. That module's
 * `formatUsd`, `formatTokens` and `formatLatency` disagree with these on
 * rounding, grouping and the absent-value spelling, so swapping one for the
 * other is a behavior change rather than a move; otari#1335 tracks it.
 */

import type { ChargeLine, UsageEntry } from "@/client"
import { isUnitChargeLine } from "@/client"
import {
  formatCost,
  formatLatency,
  formatNumber,
} from "@/shared/helpers/format"
import {
  ACTIVITY_DEFAULT_KEY,
  ACTIVITY_PRESETS,
  CUSTOM_KEY,
  findPreset,
  isoAgo,
  YEAR_SPAN_S,
} from "@/shared/helpers/timeRange"

// ---------- formatting ----------

export function formatUSD(value: number | null): string {
  return value === null ? "—" : formatCost(value)
}

// The full grouped count rather than the compact `formatTokens`: a request log
// is read for the exact number of tokens a call billed, where a tile is read
// for scale.
export function formatTokenCount(value: number | null): string {
  return value === null ? "—" : formatNumber(value)
}

// A charge line is discriminated by which rate it carries: token meters price per
// million, gateway-run tool meters price per call.

// Token lines first, tool lines after, each group keeping the order the writers
// emitted. "Billed meters" otherwise reads as an unordered mix once a row has both.
export function sortChargeLines(lines: readonly ChargeLine[]): ChargeLine[] {
  return [...lines].sort(
    (a, b) => Number(isUnitChargeLine(a)) - Number(isUnitChargeLine(b)),
  )
}

// A row that recorded no latency (historical rows, batch jobs) renders as an em
// dash so the column stays aligned, which is what the shared helper's
// `undefined` leaves each surface to decide.
export function formatLatencyCell(ms: number | null): string {
  return formatLatency(ms) ?? "—"
}

// Coarser than the settled Total time column: this is a wall-clock wait an
// operator is watching rather than a measurement, so sub-second precision is
// noise. Minutes appear because a stuck local model is the case this exists for.
export function formatElapsed(ms: number): string {
  const seconds = Math.floor(ms / 1000)
  if (seconds < 60) return `${seconds}s`
  const minutes = Math.floor(seconds / 60)
  return `${minutes}m ${String(seconds % 60).padStart(2, "0")}s`
}

// Stable row-key getter and row class so DataTable's per-row cache holds
// across re-renders (see the DataTable docstring); an inline arrow here would
// rebuild every row on each selection click.
export const getActivityRowKey = (entry: UsageEntry): string => entry.id

// An absorbed attempt is a failure a routing policy recovered from, so the
// request it belongs to succeeded. Styling it like an error would make a working
// fallback chain read as an outage, which is the same reason the server keeps it
// out of error_count. Amber says "something happened here" without saying "this
// request failed".
export const getActivityRowClassName = (
  entry: UsageEntry,
): string | undefined => {
  if (entry.status === "error") return "bg-danger-subtle"
  if (entry.status === "absorbed") return "bg-warning-subtle"
  return undefined
}

// ---------- filter option sets ----------
//
// The time presets and window math are shared with the Usage page via
// `@/shared/helpers/timeRange` (see the ActivityTimeline selector). Activity keeps a
// truthful "All": its raw list endpoint applies no default and no clamp, so an
// omitted start really is all-time.

export const STATUS_OPTIONS: { label: string; value: string }[] = [
  { label: "All", value: "" },
  { label: "Success", value: "success" },
  { label: "Error", value: "error" },
  // An attempt a routing policy recovered from. Listed because the rows are
  // rendered and styled distinctly, so an operator who spots one has to be able
  // to filter to the rest of them.
  { label: "Absorbed", value: "absorbed" },
]

export const PRICED_OPTIONS: { label: string; value: string }[] = [
  { label: "All", value: "" },
  { label: "Priced", value: "true" },
  { label: "Unpriced", value: "false" },
]

// Gateway-run tools an operator can filter on. "Any tool" also matches MCP tools,
// whose names come from the caller's own server and so cannot be enumerated here.
export const TOOL_OPTIONS: { label: string; value: string }[] = [
  { label: "All", value: "" },
  { label: "Any tool", value: "any" },
  { label: "Web search", value: "web_search" },
  { label: "Web fetch", value: "web_fetch" },
  { label: "Code execution", value: "code_execution" },
]

// Resolve the query window. Explicit start_date/end_date bounds (a custom range,
// or a drill-down from the Usage page) take precedence; otherwise a preset anchors
// `start` to "now minus N", and "all" (or an empty custom range) leaves it open.
// `now` is a parameter, not a call inside, so a caller deriving more than one
// window can hand both the same clock reading. Read independently they land
// milliseconds apart, and `winOutsideExtent` in `ActivityPage` compares two of
// them for strict inequality, so drift of a single millisecond changes what the
// page does.
export function resolveWindow(
  range: string,
  start: string,
  end: string,
  now: number = Date.now(),
): { start?: string; end?: string } {
  if (start || end) {
    return { start: start || undefined, end: end || undefined }
  }
  if (range === CUSTOM_KEY) {
    return {}
  }
  const preset =
    findPreset(ACTIVITY_PRESETS, range) ??
    findPreset(ACTIVITY_PRESETS, ACTIVITY_DEFAULT_KEY)
  const seconds = preset?.seconds ?? null
  return {
    start: seconds == null ? undefined : isoAgo(seconds, now),
    end: undefined,
  }
}

// The histogram extent (what the bars span), which is *not* always the list
// window. For bounded presets it matches `resolveWindow`. Any range with no
// rolling start of its own (the unbounded "All", or the `custom` sentinel) gets an
// explicit year-long start instead: the list genuinely omits its start there, but
// the summary endpoint would then apply a hidden 30-day default, so the bars would
// silently show a rolling month while the caption reads "All time". The explicit
// start gives a deterministic, draggable span (the axis shows exactly what it
// covers) while the list stays all-time.
export function resolveExtentWindow(
  range: string,
  now: number = Date.now(),
): { start?: string; end?: string } {
  const win = resolveWindow(range, "", "", now)
  if (win.start) return win
  const preset = findPreset(ACTIVITY_PRESETS, range)
  if (preset?.seconds == null) return { start: isoAgo(YEAR_SPAN_S, now) }
  return win
}

// The column's words, in their own casing, rather than `status.toUpperCase()`
// on the raw enum: a repeated column carries its labels in the case they are
// written in, and uppercase emphasis would fall equally on the
// successes, which is the last thing a column built to surface exceptions
// wants to draw the eye to. Unknown values still render their slug.
const STATUS_LABELS: Record<string, string> = {
  error: "Error",
  absorbed: "Absorbed",
  success: "Success",
}

export function describeStatus(status: string): string {
  return STATUS_LABELS[status] ?? status
}

// Friendly labels for known provenance sources; unknown sources render their slug.
const SOURCE_LABELS: Record<string, string> = {
  gateway: "Gateway",
  claude_code: "Claude Code",
  codex: "Codex",
}

export function describeSource(source: string): string {
  return SOURCE_LABELS[source] ?? source
}

// ---------- token composition ----------
//
// One total is the least useful number on the row: on a cached agent workload it
// is ~98% cache read, so every row shows a large, similar-looking figure. The
// composition is what varies, so the column renders the split.

export interface TokenComposition {
  // Input tokens billed at the full input rate (the prompt minus whatever was
  // served from, or written to, the cache).
  fresh: number
  cacheRead: number
  cacheWrite: number
  output: number
  total: number
}

export function readPositive(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) && value > 0
    ? value
    : 0
}

// Split a row's tokens into fresh input / cache read / cache write / output, or
// null when the row carries no usage at all (an error before the provider replied).
//
// `billing_meters` is preferred because it is the *normalized* view: providers
// disagree on whether cache reads and writes are counted inside `prompt_tokens`
// (OpenAI: yes, Anthropic: no), and the row does not record which convention its
// numbers follow, so the raw columns alone cannot be split reliably. The writers
// resolve that when they price a row, so meters are present for any priced row.
// Failing that, assume the subset convention and clamp, which is exact whenever
// there is no cache usage to misattribute.
export function buildTokenComposition(
  entry: UsageEntry,
): TokenComposition | null {
  // Per key, not per object: a row can carry meters for its gateway-run tool calls
  // while its tokens were never metered (an unpriced model still owes for the
  // searches it ran). Keying off the object's presence would then read every token
  // as 0 and the bar would vanish from a row that has real tokens.
  const meters = entry.billing_meters ?? null
  const meter = (key: string, fallback: number | null): number =>
    meters && typeof meters[key] === "number"
      ? readPositive(meters[key])
      : readPositive(fallback)
  const totalInput = meter("total_input_tokens", entry.prompt_tokens)
  const cacheRead = meter("cache_read_tokens", entry.cache_read_tokens)
  const cacheWrite = meter("cache_write_tokens", entry.cache_write_tokens)
  const output = meter("completion_tokens", entry.completion_tokens)
  const fresh = Math.max(0, totalInput - cacheRead - cacheWrite)
  const total = fresh + cacheRead + cacheWrite + output
  return total > 0 ? { fresh, cacheRead, cacheWrite, output, total } : null
}

// A row's gateway-run tool calls, read out of the reserved `tools` meter namespace.
// Nested under one key so a caller-named MCP tool can never collide with a token
// meter (which the billed-token SQL and `TokenBar` both read by name).
export type ToolUsage = {
  tool: string
  billed: number
  errors: number
  unitRate: number | null
}

export function listToolUsage(entry: UsageEntry): ToolUsage[] {
  const nested = entry.billing_meters?.tools
  if (!nested || typeof nested !== "object") return []
  return Object.entries(nested as Record<string, unknown>)
    .flatMap(([tool, counts]) => {
      if (!counts || typeof counts !== "object") return []
      const record = counts as Record<string, unknown>
      const billed = readPositive(record.billed)
      const errors = readPositive(record.errors)
      if (!billed && !errors) return []
      const rate = record.unit_rate
      return [
        {
          tool,
          billed,
          errors,
          unitRate: typeof rate === "number" ? rate : null,
        },
      ]
    })
    .sort((a, b) => b.billed - a.billed || a.tool.localeCompare(b.tool))
}

// "web search x3" / "web search x3, 1 failed". The tool name is de-underscored for
// reading; failures are named rather than folded into the count, because a failed
// call is not billed and an operator chasing a cost needs that distinction.
export function formatToolUsage(usage: ToolUsage): string {
  const label = usage.tool.replaceAll("_", " ")
  const parts = usage.billed ? [`${label} \u00d7${usage.billed}`] : [label]
  if (usage.errors) parts.push(`${usage.errors} failed`)
  return parts.join(", ")
}

// Cost attributable to gateway-run tools on this row, from the rate stored with the
// row rather than the live price, so a historical row reads as it was billed.
export function computeToolCost(entry: UsageEntry): number | null {
  const usages = listToolUsage(entry).filter((usage) => usage.billed > 0)
  if (usages.some((usage) => usage.unitRate === null)) return null
  return usages.reduce(
    (sum, usage) => sum + usage.billed * (usage.unitRate ?? 0),
    0,
  )
}

// Segment order runs input side first (fresh, then the two cache buckets), then
// output. Shading is one hue at four lightnesses, assigned for legibility rather
// than for price: every fill clears the track it sits on, adjacent fills differ
// enough to show their boundary, and the bucket that is usually the bulk (cache
// read) takes a mid tone instead of the palest step, so a cache-heavy row reads
// as a filled bar and a fresh-input row as a dark one. Nothing is encoded by hue,
// and the tooltip / accessible name carry every number, so the bar adds a shape
// to scan and removes no information.
export const TOKEN_SEGMENTS: {
  key: keyof Omit<TokenComposition, "total">
  label: string
  fill: string
}[] = [
  { key: "fresh", label: "Fresh input", fill: "var(--color-chart-ramp-1)" },
  { key: "cacheRead", label: "Cache read", fill: "var(--color-chart-ramp-3)" },
  {
    key: "cacheWrite",
    label: "Cache write",
    fill: "var(--color-chart-ramp-4)",
  },
  { key: "output", label: "Output", fill: "var(--color-chart-ramp-2)" },
]

// The pricing key a usage row bills against. A row stores the instance and the
// bare model separately (`log_usage` is called with `provider=resolved.instance,
// model=resolved.model`, and a gateway rejection logs the same pair), while
// pricing is looked up as `instance:model` (`find_model_pricing`), so the key has
// to be rebuilt from both: `entry.model` alone is prefix-less and would store a
// price nothing ever reads. A row whose selector never resolved carries no
// provider and its model is the raw selector, so that is used as-is.
export function findPricingSelector(entry: UsageEntry): string {
  if (!entry.provider) return entry.model
  return entry.model.startsWith(`${entry.provider}:`)
    ? entry.model
    : `${entry.provider}:${entry.model}`
}

// ---------- routing ----------
//
// A routed request writes one usage row per attempt (all sharing a
// `request_group_id`), so a single row answers only half of what an operator
// wants: "attempt 1 of 2 failed" without saying what served the request. These
// helpers turn the stored attribution into the sentence the Routing column shows,
// and reassemble a plan from the rows of one group.

// Plain English for a compiled attempt's `selection_reason`. The stored values are
// the compiler's vocabulary (`static`, `default`, `on_failure`, `condition:<keys>`,
// `router:<name>`): precise, and meaningless to a reader who has not read the
// compiler. An unrecognized value is de-underscored rather than dropped, since the
// set is open by construction (a condition or router name comes from config).
export function describeSelectionReason(
  reason: string | null | undefined,
): string | null {
  if (!reason) return null
  if (reason === "static") return "the policy's only target"
  if (reason === "default") return "the policy's default target"
  if (reason === "on_failure") return "a fallback candidate"
  if (reason.startsWith("condition:")) {
    const keys = reason
      .slice("condition:".length)
      .split(",")
      .filter(Boolean)
      .join(", ")
    return keys ? `matched on ${keys}` : "matched a condition"
  }
  if (reason.startsWith("router:")) {
    const name = reason.slice("router:".length)
    return name ? `chosen by router ${name}` : "chosen by a router"
  }
  return reason.replaceAll("_", " ")
}

// What a group's request ended up doing, read off its outcome row. Absorbed rows
// are attempts the policy recovered from, so exactly one row per finished group is
// the outcome: the attempt that served, or the terminal failure.
export interface GroupOutcome {
  /** Qualified target of the attempt that served, or null when none did. */
  servedBy: string | null
  servedPosition: number | null
}

// Index the outcome of every group represented in `rows`. Built from rows the page
// already holds first, then filled in from a batched lookup, so the common case
// (a group's attempts are adjacent in a newest-first list) costs no extra request.
export function indexGroupOutcomes(
  rows: readonly UsageEntry[],
): Map<string, GroupOutcome> {
  return new Map(
    rows.flatMap((row): [string, GroupOutcome][] =>
      row.request_group_id && row.status !== "absorbed"
        ? [
            [
              row.request_group_id,
              {
                servedBy:
                  row.status === "success" ? findPricingSelector(row) : null,
                servedPosition:
                  row.status === "success"
                    ? (row.attempt_position ?? null)
                    : null,
              },
            ],
          ]
        : [],
    ),
  )
}

// One line of prose for a row's place in its plan, replacing the "attempt 1/2 ·
// default" shorthand: that read as a fraction of something unnamed, said nothing
// about whether the attempt worked, and pointed at no other row. `outcome` is the
// group's outcome when it is known, which is what lets an absorbed row name the
// model that served in its place.
export function describeAttempt(
  entry: UsageEntry,
  outcome: GroupOutcome | null,
): string | null {
  const reason = describeSelectionReason(entry.selection_reason)
  const position = entry.attempt_position
  const total = entry.attempt_count
  // A policy with one candidate has no plan to place the row in, so the only thing
  // worth saying is why that candidate was picked.
  if (position == null || total == null || total <= 1) return reason
  const attempt = `attempt ${position} of ${total}`
  if (entry.status === "absorbed") {
    if (outcome?.servedBy)
      return `${attempt} failed, served by ${outcome.servedBy}`
    // Not "and so did the rest": the group's outcome row is an error, but the walk
    // may have stopped on it (a non-retryable status, a lock-in) with later
    // candidates never called, which is what that row's own sentence says.
    if (outcome) return `${attempt} failed, and the request ended in an error`
    return `${attempt} failed, fell back`
  }
  if (entry.status === "error") {
    // The walk stops early on a non-retryable failure, a lock-in, or a
    // gateway-side refusal, so the later candidates were not necessarily tried.
    return position < total
      ? `${attempt} failed, no further candidate tried`
      : `${attempt} failed, plan exhausted`
  }
  return reason ? `served on ${attempt} (${reason})` : `served on ${attempt}`
}

// Per-attempt outcome for the plan table. Terser than the row sentence, which has
// to stand alone; here the table's shape already says which attempt this is.
export function describeAttemptOutcome(entry: UsageEntry): string {
  if (entry.status === "absorbed")
    return entry.status_code === null
      ? "failed, fell back"
      : `failed ${entry.status_code}, fell back`
  if (entry.status === "error")
    return entry.status_code === null ? "failed" : `failed ${entry.status_code}`
  return "served the request"
}

// Attempts in plan order. `attempt_position` is authoritative; timestamp is the
// tiebreaker for a row written before the column existed, or a group whose rows
// share a position (which would be a writer bug, not something to hide).
export function sortPlanRows(rows: readonly UsageEntry[]): UsageEntry[] {
  return [...rows].sort(
    (a, b) =>
      (a.attempt_position ?? 0) - (b.attempt_position ?? 0) ||
      a.timestamp.localeCompare(b.timestamp),
  )
}
