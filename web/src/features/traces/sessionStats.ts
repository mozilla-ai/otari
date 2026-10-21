import type { TraceDetail, TraceSpan, TraceSummary } from "@/client"

import { type GuardrailVerdict, gateVerdict } from "./gateway"

export interface ToolTally {
  tool: string
  calls: number
  /** Time its calls spent running, overlapping calls counted once. */
  runningMs: number
}

export interface ModelTally {
  model: string
  requests: number
  /** What the gateway recorded for its requests; null when none of them carried a cost. */
  cost: number | null
}

export interface Meters {
  input: number
  output: number
  cacheRead: number
  cacheWrite: number
}

export interface SessionStats {
  turns: number
  lengthMs: number
  requests: number
  toolCalls: number
  inSubagents: number
  tools: ToolTally[]
  /** Each agent guardrail gate a hook ran, by what it decided. */
  gates: { runs: number; verdicts: Partial<Record<GuardrailVerdict, number>> }
  /** The judge gates that called a model: what they ran on and what the priced ones cost. */
  judges: { calls: number; models: string[]; cost: number; priced: number }
  /**
   * Each overlapping span counted once. A tool runs until any subagent it started
   * finishes, and that subagent's requests count as model time too, so these can
   * add up past `lengthMs`.
   */
  time: { modelMs: number; toolsMs: number; waitingMs: number; hooksMs: number }
  /** Time something was happening, overlaps counted once; the rest of `lengthMs` was idle. */
  activeMs: number
  /** Tool time spent wholly inside a model request, which is where a gateway's built-in tools run. */
  toolsInModelMs: number
  tokens: Meters
  /** What the priced requests cost per meter; null when none was priced by the token. */
  cost: Meters | null
  /** What reading from the cache saved over paying for those tokens as fresh input. */
  cacheSaved: number | null
  pricedRequests: number
  /** Each model the requests ran on, the costliest first. */
  models: ModelTally[]
}

const EXECUTION = "tool.execution"
const WAITING = "tool.blocked_on_user"
// Claude Code's way of handing a subagent's answer back, not a tool the agent chose.
const NOT_A_TOOL = new Set(["SubagentHandback"])
const METERS = ["input", "output", "cache_read", "cache_write"] as const

function interval(span: TraceSpan): [number, number] | null {
  if (!span.start_time || !span.end_time) return null
  const start = Date.parse(span.start_time)
  const end = Date.parse(span.end_time)
  return end > start ? [start, end] : null
}

type Interval = [number, number]

// The intervals as the time they cover: sorted, with every overlap merged.
function merged(intervals: Array<Interval | null>): Interval[] {
  const sorted = intervals
    .filter((i): i is Interval => i !== null)
    .sort((a, b) => a[0] - b[0])
  const out: Interval[] = []
  for (const [start, end] of sorted) {
    const last = out.at(-1)
    if (last && start <= last[1]) last[1] = Math.max(last[1], end)
    else out.push([start, end])
  }
  return out
}

function unionMs(intervals: Array<Interval | null>): number {
  return merged(intervals).reduce(
    (total, [start, end]) => total + end - start,
    0,
  )
}

function isWithin(inner: Interval, cover: Interval[]): boolean {
  return cover.some(([start, end]) => start <= inner[0] && inner[1] <= end)
}

function spansMs(spans: readonly TraceSpan[]): number {
  return unionMs(spans.map(interval))
}

function rates(
  span: TraceSpan,
): Record<(typeof METERS)[number], number> | null {
  const out: Partial<Record<(typeof METERS)[number], number>> = {}
  for (const meter of METERS) {
    const value = span.attributes[`otari.llm.rate.${meter}`]
    const rate = typeof value === "string" ? Number(value) : Number.NaN
    if (!Number.isFinite(rate)) return null
    out[meter] = rate / 1_000_000
  }
  return out as Record<(typeof METERS)[number], number>
}

// A session at a glance: what it did, where its time went and what its money
// paid for, read off its spans.
export function sessionStats(
  summary: TraceSummary,
  detail: TraceDetail,
): SessionStats {
  const byId = new Map(detail.spans.map((span) => [span.span_id, span]))
  const inSubagent = (span: TraceSpan): boolean => {
    // Parent ids are as the exporter sent them, so a walk guards against a cycle.
    const seen = new Set([span.span_id])
    let parent = span.parent_span_id ? byId.get(span.parent_span_id) : undefined
    while (parent && !seen.has(parent.span_id)) {
      seen.add(parent.span_id)
      if (parent.operation === EXECUTION) return true
      parent = parent.parent_span_id
        ? byId.get(parent.parent_span_id)
        : undefined
    }
    return false
  }
  const tools = detail.spans.filter(
    (span) => span.kind === "tool" && !NOT_A_TOOL.has(span.tool_name ?? ""),
  )
  const llm = detail.spans.filter((span) => span.kind === "llm")
  const runs = new Map<string, TraceSpan[]>()
  for (const span of detail.spans) {
    if (span.operation === EXECUTION && span.parent_span_id) {
      runs.set(span.parent_span_id, [
        ...(runs.get(span.parent_span_id) ?? []),
        span,
      ])
    }
  }
  const children = new Map<string, TraceSpan[]>()
  for (const span of detail.spans) {
    if (!span.parent_span_id) continue
    children.set(span.parent_span_id, [
      ...(children.get(span.parent_span_id) ?? []),
      span,
    ])
  }
  const lastEnd = (span: TraceSpan, seen = new Set<string>()): number => {
    if (seen.has(span.span_id)) return 0
    seen.add(span.span_id)
    let end = span.end_time ? Date.parse(span.end_time) : 0
    for (const child of children.get(span.span_id) ?? []) {
      end = Math.max(end, lastEnd(child, seen))
    }
    return end
  }
  const tallies = new Map<
    string,
    { calls: number; intervals: Array<[number, number] | null> }
  >()
  // A model request's span, by the span it ran under.
  const requestsUnder = new Map<string, Interval[]>()
  for (const span of llm) {
    const own = interval(span)
    if (!own) continue
    const key = span.parent_span_id ?? ""
    const under = requestsUnder.get(key)
    if (under) under.push(own)
    else requestsUnder.set(key, [own])
  }
  for (const [key, list] of requestsUnder) requestsUnder.set(key, merged(list))
  const running: Interval[] = []
  const inRequests: Interval[] = []
  for (const span of tools) {
    const name = span.tool_name ?? "tool"
    const tally = tallies.get(name) ?? { calls: 0, intervals: [] }
    tally.calls += 1
    // Its run, when the agent reported one, leaves out waiting for approval. A
    // run that started a subagent ends when the subagent's last span does,
    // since the run's own span closes as soon as it is started.
    const cover = requestsUnder.get(span.parent_span_id ?? "") ?? []
    for (const run of runs.get(span.span_id) ?? [span]) {
      const own = interval(run)
      if (!own) continue
      const ran: Interval = [own[0], Math.max(own[1], lastEnd(run))]
      tally.intervals.push(ran)
      running.push(ran)
      // Wholly inside a request it ran beside, as a gateway's built-in tool
      // runs inside the request that called it. A subagent is the reverse: its
      // requests run inside the tool, and under the tool rather than beside it.
      if (isWithin(ran, cover)) inRequests.push(ran)
    }
    tallies.set(name, tally)
  }
  const tokens: Meters = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }
  const cost: Meters = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }
  let cacheSaved = 0
  let pricedRequests = 0
  const models = new Map<string, ModelTally>()
  for (const span of llm) {
    const model = span.model ?? "unknown model"
    const tally = models.get(model) ?? { model, requests: 0, cost: null }
    tally.requests += 1
    if (span.cost != null) tally.cost = (tally.cost ?? 0) + span.cost
    models.set(model, tally)
    const cacheRead = span.cache_read_tokens ?? 0
    const cacheWrite = span.cache_write_tokens ?? 0
    const used: Meters = {
      // A span's input tokens count the cache in; the input meter is the rest.
      input: Math.max(0, (span.input_tokens ?? 0) - cacheRead - cacheWrite),
      output: span.output_tokens ?? 0,
      cacheRead,
      cacheWrite,
    }
    tokens.input += used.input
    tokens.output += used.output
    tokens.cacheRead += used.cacheRead
    tokens.cacheWrite += used.cacheWrite
    const rate = rates(span)
    if (!rate) continue
    pricedRequests += 1
    cost.input += used.input * rate.input
    cost.output += used.output * rate.output
    cost.cacheRead += used.cacheRead * rate.cache_read
    cost.cacheWrite += used.cacheWrite * rate.cache_write
    cacheSaved += used.cacheRead * (rate.input - rate.cache_read)
  }
  const waits = detail.spans.filter((span) => span.operation === WAITING)
  // A gateway guardrail is a guardrail span too, but not a hook's.
  const hooks = detail.spans.filter((span) => gateVerdict(span))
  const gates: SessionStats["gates"] = { runs: 0, verdicts: {} }
  const judges: SessionStats["judges"] = {
    calls: 0,
    models: [],
    cost: 0,
    priced: 0,
  }
  for (const span of detail.spans) {
    const verdict = gateVerdict(span)
    if (!verdict) continue
    gates.runs += 1
    gates.verdicts[verdict] = (gates.verdicts[verdict] ?? 0) + 1
    if (!span.model) continue
    judges.calls += 1
    if (!judges.models.includes(span.model)) judges.models.push(span.model)
    if (span.cost != null) {
      judges.cost += span.cost
      judges.priced += 1
    }
  }
  const activeMs = unionMs([
    ...running,
    ...[...llm, ...waits, ...hooks].map(interval),
  ])
  return {
    turns: detail.turns.length,
    // The spans can outlast the summary's clock by a little, so the session
    // is at least as long as what happened in it.
    lengthMs: Math.max(
      activeMs,
      Date.parse(summary.last_activity_at) - Date.parse(summary.started_at),
    ),
    requests: llm.length,
    toolCalls: tools.length,
    inSubagents: tools.filter(inSubagent).length,
    tools: [...tallies]
      .map(([tool, tally]) => ({
        tool,
        calls: tally.calls,
        runningMs: unionMs(tally.intervals),
      }))
      .sort((a, b) => b.calls - a.calls || b.runningMs - a.runningMs),
    gates,
    judges,
    time: {
      modelMs: spansMs(llm),
      toolsMs: unionMs(running),
      waitingMs: spansMs(waits),
      hooksMs: spansMs(hooks),
    },
    activeMs,
    toolsInModelMs: unionMs(inRequests),
    tokens,
    cost: pricedRequests > 0 ? cost : null,
    cacheSaved: pricedRequests > 0 ? cacheSaved : null,
    pricedRequests,
    models: [...models.values()].sort((a, b) => (b.cost ?? 0) - (a.cost ?? 0)),
  }
}
