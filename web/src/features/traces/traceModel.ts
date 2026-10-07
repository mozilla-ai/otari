import type { TraceDetail, TraceSpan, TraceSummary, TraceTurn } from "@/client"
import type { ChipTone } from "@/design-system/indicators/Chip"

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
      return "Guardrail check"
    case "mcp_connect":
      return "MCP connection"
    case "agent":
      return span.name
    default:
      return span.name
  }
}

export function spanTone(span: TraceSpan): ChipTone {
  if (span.outcome === "error") {
    return span.recovered ? "warning" : "danger"
  }
  return span.outcome === "unknown" ? "neutral" : "success"
}

export function spanOutcomeLabel(span: TraceSpan): string {
  if (span.outcome === "error") {
    return span.recovered ? "Recovered" : "Failed"
  }
  return span.outcome === "unknown" ? "No result" : "OK"
}

const TURN_STATE_TONE: Record<TraceTurn["state"], ChipTone> = {
  active: "info",
  completed: "success",
  failed: "danger",
  incomplete: "warning",
}

const TURN_STATE_LABEL: Record<TraceTurn["state"], string> = {
  active: "Active",
  completed: "Completed",
  failed: "Failed",
  incomplete: "Incomplete",
}

export function turnStatus(turn: TraceTurn): { tone: ChipTone; label: string } {
  return {
    tone: TURN_STATE_TONE[turn.state],
    label: TURN_STATE_LABEL[turn.state],
  }
}

// What a session is called in the list: the agent that ran it, and a short id
// to tell two runs of one agent apart.
export function sessionName(summary: TraceSummary): string {
  return (
    summary.harness ??
    (summary.session_source === "none" ? "Request" : "Session")
  )
}

export function shortId(traceId: string): string {
  return traceId.replace(/^s-/, "").slice(0, 10)
}

export function sessionFailed(summary: TraceSummary): boolean {
  return summary.error_count > 0
}

const SOURCE_LABEL: Record<TraceSummary["session_source"], string> = {
  client: "Grouped by the client's session id",
  harness: "Grouped by the agent's own session id",
  otlp: "Exported by an instrumented agent",
  none: "A single request: the client named no session",
}

export function sessionSourceLabel(summary: TraceSummary): string {
  return SOURCE_LABEL[summary.session_source]
}

// The flat view: every span in start order, for reading a session as a log.
export function spanLog(detail: TraceDetail): TraceSpan[] {
  return [...detail.spans].sort(byStart)
}

export function sessionDurationMs(summary: TraceSummary): number {
  return Math.max(
    0,
    Date.parse(summary.last_activity_at) - Date.parse(summary.started_at),
  )
}
