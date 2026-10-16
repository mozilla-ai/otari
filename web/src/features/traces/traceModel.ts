import type { TraceDetail, TraceSpan, TraceSummary, TraceTurn } from "@/client"

import { guardrailProfile, mcpServer } from "./gateway"

export interface SpanNode {
  span: TraceSpan
  children: SpanNode[]
}

export interface TurnNode {
  turn: TraceTurn
  steps: SpanNode[]
}

// A session as the tree draws it: each turn with its steps and everything under
// them, plus the spans that hang under no step (an instrumented agent's own
// root spans), which keep the tree they were exported with.
export interface TraceTree {
  turns: TurnNode[]
  roots: SpanNode[]
}

function byStart(a: TraceSpan, b: TraceSpan): number {
  return (a.start_time ?? a.end_time ?? "").localeCompare(
    b.start_time ?? b.end_time ?? "",
  )
}

export function buildTraceTree(detail: TraceDetail): TraceTree {
  const children = new Map<string, TraceSpan[]>()
  const ids = new Set(detail.spans.map((span) => span.span_id))
  for (const span of detail.spans) {
    if (span.parent_span_id && ids.has(span.parent_span_id)) {
      const siblings = children.get(span.parent_span_id) ?? []
      siblings.push(span)
      children.set(span.parent_span_id, siblings)
    }
  }
  const node = (span: TraceSpan): SpanNode => ({
    span,
    children: [...(children.get(span.span_id) ?? [])].sort(byStart).map(node),
  })
  const byId = new Map(detail.spans.map((span) => [span.span_id, span]))
  const turns = detail.turns.map((turn) => ({
    turn,
    steps: turn.step_ids.flatMap((id) => {
      const step = byId.get(id)
      return step ? [node(step)] : []
    }),
  }))
  const roots = detail.spans
    .filter(
      (span) =>
        span.kind !== "step" &&
        !(span.parent_span_id && ids.has(span.parent_span_id)),
    )
    .sort(byStart)
    .map(node)
  return { turns, roots }
}

// What a span is called in the tree: the thing it acted on where there is one
// (the model, the tool), and its kind where there is not.
export function spanLabel(span: TraceSpan): string {
  switch (span.kind) {
    case "llm":
      return span.model || "LLM call"
    case "tool":
      return span.tool_name || "Tool call"
    case "step":
      return "Request"
    case "routing_attempt":
      return "Routing attempt"
    case "guardrail":
      return guardrailProfile(span) ?? "Guardrail check"
    case "mcp_connect":
      return mcpServer(span) ?? "MCP connection"
    case "agent":
      return span.name
    default:
      return span.name
  }
}

// A row of the activity list is a session (requests that named one) or a single
// request that named none; both are traces, so they share one list.
export function traceKind(summary: TraceSummary): "Session" | "Request" {
  return summary.session_source === "none" ? "Request" : "Session"
}

// What a session is called in the list: the agent that ran it, and a short id
// to tell two runs of one agent apart.
export function sessionName(summary: TraceSummary): string {
  return summary.harness ?? traceKind(summary)
}

export function shortId(traceId: string): string {
  return traceId.replace(/^s-/, "").slice(0, 10)
}

export function sessionDurationMs(summary: TraceSummary): number {
  return Math.max(
    0,
    Date.parse(summary.last_activity_at) - Date.parse(summary.started_at),
  )
}
