import type { TraceDetail, TraceSpan } from "@/client"

export interface ToolCostRow {
  tool: string
  calls: number
  cost: number
  /** The dearest of its calls, the one to open first. */
  spanId: string
  /** The models its calls' subagents ran on, most used first. */
  models: string[]
  /** The calls its subagents made, already counted in `cost`. */
  inside: ToolCostRow[]
  /** What its subagents' own requests cost beyond their tool calls: their reasoning and answers. */
  ownRequests: number
}

export interface ToolCostBreakdown {
  rows: ToolCostRow[]
  /** What no tool call accounts for: the prompts, the system prompt and the answers. */
  untied: number
  /** What judge gates' model calls cost, which no tool call made either. */
  judges: number
}

const EXECUTION = "tool.execution"

/** The run of the subagent call a span sits under, if any: the outermost, so a nested subagent counts once. */
function outermostRun(
  span: TraceSpan,
  byId: ReadonlyMap<string, TraceSpan>,
): TraceSpan | null {
  let run: TraceSpan | null = null
  // Parent ids are as the exporter sent them, so the walk guards against a cycle.
  const seen = new Set([span.span_id])
  let parent = span.parent_span_id ? byId.get(span.parent_span_id) : undefined
  while (parent && !seen.has(parent.span_id)) {
    seen.add(parent.span_id)
    if (parent.operation === EXECUTION) run = parent
    parent = parent.parent_span_id ? byId.get(parent.parent_span_id) : undefined
  }
  return run
}

function byCost(rows: Map<string, ToolCostRow>): ToolCostRow[] {
  return [...rows.values()].sort((a, b) => b.cost - a.cost)
}

function add(
  rows: Map<string, ToolCostRow>,
  tool: string,
  spanId: string,
  cost: number,
  dearest: Map<string, number>,
): ToolCostRow {
  const row = rows.get(tool) ?? {
    tool,
    calls: 0,
    cost: 0,
    spanId,
    models: [],
    inside: [],
    ownRequests: 0,
  }
  row.calls += 1
  row.cost += cost
  if (cost > (dearest.get(row.spanId) ?? Number.NEGATIVE_INFINITY)) {
    row.spanId = spanId
  }
  dearest.set(spanId, cost)
  rows.set(tool, row)
  return row
}

// What each tool cost across a session, from the estimate on each call. A call
// a subagent made is part of the exact cost of the call that ran the subagent,
// so it is listed under that call rather than counted again.
export function toolCostBreakdown(
  detail: TraceDetail,
  sessionCost: number,
): ToolCostBreakdown | null {
  if (detail.tool_costs.length === 0) return null
  const byId = new Map(detail.spans.map((span) => [span.span_id, span]))
  const dearest = new Map<string, number>()
  const top = new Map<string, ToolCostRow>()
  const inside = new Map<string, Map<string, ToolCostRow>>()
  const subagentCost = new Map<string, number>()
  for (const estimate of detail.tool_costs) {
    const span = byId.get(estimate.span_id)
    if (!span) continue
    const tool = span.tool_name ?? "tool"
    const run = outermostRun(span, byId)
    const owner = run?.parent_span_id ? byId.get(run.parent_span_id) : null
    if (owner) {
      const ownerTool = owner.tool_name ?? "tool"
      const rows = inside.get(ownerTool) ?? new Map<string, ToolCostRow>()
      // Its own subagent's cost, if any, is inside the outer one's already.
      add(
        rows,
        tool,
        span.span_id,
        estimate.total - (estimate.subagent ?? 0),
        dearest,
      )
      inside.set(ownerTool, rows)
    } else {
      add(top, tool, span.span_id, estimate.total, dearest)
      if (estimate.subagent != null) {
        subagentCost.set(
          tool,
          (subagentCost.get(tool) ?? 0) + estimate.subagent,
        )
      }
    }
  }
  const models = new Map<string, Map<string, number>>()
  for (const span of detail.spans) {
    if (span.kind !== "llm" || !span.model) continue
    const run = outermostRun(span, byId)
    const owner = run?.parent_span_id ? byId.get(run.parent_span_id) : null
    if (!owner) continue
    const tool = owner.tool_name ?? "tool"
    const counts = models.get(tool) ?? new Map<string, number>()
    counts.set(span.model, (counts.get(span.model) ?? 0) + 1)
    models.set(tool, counts)
  }
  const rows = byCost(top)
  for (const row of rows) {
    row.inside = byCost(inside.get(row.tool) ?? new Map())
    const insideCost = row.inside.reduce((sum, r) => sum + r.cost, 0)
    row.ownRequests = Math.max(
      0,
      (subagentCost.get(row.tool) ?? 0) - insideCost,
    )
    row.models = [...(models.get(row.tool) ?? new Map<string, number>())]
      .sort((a, b) => b[1] - a[1])
      .map(([model]) => model)
  }
  const tied = rows.reduce((sum, row) => sum + row.cost, 0)
  const judges = detail.spans.reduce(
    (sum, span) =>
      span.kind === "guardrail" && span.model ? sum + (span.cost ?? 0) : sum,
    0,
  )
  return { rows, untied: Math.max(0, sessionCost - tied - judges), judges }
}
