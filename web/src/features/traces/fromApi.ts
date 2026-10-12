import type {
  TraceDetail,
  TraceSpan,
  TraceSummary as TraceSummaryRecord,
  TraceTurn,
} from "@/client"

import {
  buildTraceTree,
  type SpanNode,
  sessionName,
  sessionSourceLabel,
  spanLabel,
  traceKind,
} from "./traceModel"
import type {
  Observation,
  ObservationLevel,
  ObservationType,
  Trace,
} from "./types"

const TYPE_BY_KIND: Record<string, ObservationType> = {
  llm: "GENERATION",
  tool: "TOOL",
  agent: "AGENT",
}

const TURN_LEVEL: Record<TraceTurn["state"], ObservationLevel> = {
  active: "DEFAULT",
  completed: "DEFAULT",
  failed: "ERROR",
  incomplete: "WARNING",
}

function spanLevel(span: TraceSpan): ObservationLevel {
  if (span.outcome !== "error") return "DEFAULT"
  return span.recovered ? "WARNING" : "ERROR"
}

function spanStatus(span: TraceSpan): string | undefined {
  if (span.outcome !== "error") return undefined
  const error = span.error_class ?? "error"
  return span.recovered ? `${error}, recovered by a fallback` : error
}

function spanMetadata(span: TraceSpan): Record<string, string> {
  const metadata: Record<string, string> = { Kind: span.kind }
  if (span.operation) metadata.Operation = span.operation
  metadata.Origin = span.origin
  if (span.tool_type) metadata["Tool type"] = span.tool_type
  if (span.tool_call_id) metadata["Tool call ID"] = span.tool_call_id
  if (span.request_id) metadata["Request ID"] = span.request_id
  // A client-run tool's start is inferred from the step before it.
  if (span.approximate) metadata.Timing = "Approximate"
  for (const [key, value] of Object.entries(span.attributes)) {
    metadata[key] = String(value)
  }
  return metadata
}

// A session as the explorer draws it. Without `detail` it is the list row alone,
// so the peek can open on the summary while the spans load. Turns become parent
// observations over their steps once a session has more than one, the grouping
// `buildTraceTree` reads from the API; a single request is not wrapped in one.
export function toTrace(
  summary: TraceSummaryRecord,
  detail?: TraceDetail,
): Trace {
  const origin = Date.parse(summary.started_at)
  const offset = (iso: string | null) =>
    iso == null ? null : Math.max(0, Date.parse(iso) - origin)
  const observations: Observation[] = []

  if (detail) {
    const tree = buildTraceTree(detail)
    const turnOf = new Map<string, string>()
    if (tree.turns.length > 1) {
      for (const { turn, steps } of tree.turns) {
        const id = `turn-${turn.index}`
        for (const step of steps) turnOf.set(step.span.span_id, id)
        const starts = steps.map((s) => offset(s.span.start_time) ?? 0)
        const startMs =
          offset(turn.started_at) ?? (starts.length ? Math.min(...starts) : 0)
        const endMs = offset(turn.ended_at) ?? startMs
        observations.push({
          id,
          traceId: summary.trace_id,
          parentObservationId: null,
          type: "SPAN",
          name: turn.continued ? "Earlier steps" : `Turn ${turn.index + 1}`,
          startMs,
          endMs: Math.max(startMs, endMs),
          level: TURN_LEVEL[turn.state],
          statusMessage:
            turn.state === "failed" || turn.state === "incomplete"
              ? `Turn ${turn.state}`
              : undefined,
          promptTokens: turn.input_tokens,
          completionTokens: turn.output_tokens,
          metadata: {
            State: turn.state,
            "LLM calls": String(turn.llm_calls),
            "Tool calls": String(turn.tool_calls),
          },
        })
      }
    }

    const ids = new Set(detail.spans.map((span) => span.span_id))
    const visit = (node: SpanNode) => {
      const { span } = node
      const startMs = offset(span.start_time ?? span.end_time) ?? 0
      const endMs = offset(span.end_time) ?? startMs + (span.duration_ms ?? 0)
      observations.push({
        id: span.span_id,
        traceId: summary.trace_id,
        parentObservationId:
          span.parent_span_id && ids.has(span.parent_span_id)
            ? span.parent_span_id
            : (turnOf.get(span.span_id) ?? null),
        type: TYPE_BY_KIND[span.kind] ?? "SPAN",
        name: spanLabel(span),
        startMs,
        endMs: Math.max(startMs, endMs),
        level: spanLevel(span),
        statusMessage: spanStatus(span),
        model: span.model ?? undefined,
        provider: span.provider ?? undefined,
        promptTokens: span.input_tokens ?? undefined,
        completionTokens: span.output_tokens ?? undefined,
        cost: span.cost ?? undefined,
        metadata: spanMetadata(span),
      })
      for (const child of node.children) visit(child)
    }
    for (const { steps } of tree.turns) for (const step of steps) visit(step)
    for (const root of tree.roots) visit(root)
  }

  return {
    id: summary.trace_id,
    name: sessionName(summary),
    timestamp: summary.started_at,
    userId: summary.user_id ?? "",
    sessionId: summary.trace_id,
    environment: "",
    tags: [traceKind(summary)],
    input: null,
    output: null,
    metadata: {
      Source: sessionSourceLabel(summary),
      Workspace: summary.workspace_id,
      ...(detail ? { State: detail.state } : {}),
    },
    observations,
  }
}
