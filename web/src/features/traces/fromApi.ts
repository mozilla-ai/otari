import type { TraceDetail, TraceSpan, TraceSummary, TraceTurn } from "@/client"

import {
  failureLabel,
  type GuardrailVerdict,
  gatewayEvents,
  guardrailProfile,
  guardrailVerdict,
  mcpServer,
} from "./gateway"
import {
  buildTraceTree,
  type SpanNode,
  sessionName,
  spanLabel,
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
  guardrail: "GUARDRAIL",
  mcp_connect: "MCP",
}

const TURN_LEVEL: Record<TraceTurn["state"], ObservationLevel> = {
  active: "DEFAULT",
  completed: "DEFAULT",
  failed: "ERROR",
  incomplete: "WARNING",
}

const GUARDRAIL_LEVEL: Record<GuardrailVerdict, ObservationLevel> = {
  passed: "DEFAULT",
  blocked: "ERROR",
  flagged: "WARNING",
  refused: "ERROR",
  unchecked: "WARNING",
}

const GUARDRAIL_STATUS: Record<GuardrailVerdict, string | undefined> = {
  passed: undefined,
  blocked: "Flagged, and the request was refused.",
  flagged: "Flagged in monitor mode. The request still went to the model.",
  refused: "Could not run, and the request was refused.",
  unchecked: "Could not run. The request went to the model unchecked.",
}

function spanLevel(
  span: TraceSpan,
  verdict: GuardrailVerdict | null,
): ObservationLevel {
  if (verdict) return GUARDRAIL_LEVEL[verdict]
  if (span.outcome !== "error") return "DEFAULT"
  return span.recovered ? "WARNING" : "ERROR"
}

function spanStatus(
  span: TraceSpan,
  verdict: GuardrailVerdict | null,
): string | undefined {
  if (verdict) return GUARDRAIL_STATUS[verdict]
  if (span.outcome !== "error") return undefined
  if (span.kind === "mcp_connect") {
    return "Connecting to the server or listing its tools failed."
  }
  const error = failureLabel(span.error_class)
  return span.recovered ? `${error}, recovered by a fallback` : error
}

function spanMetadata(span: TraceSpan): Record<string, string> {
  const metadata: Record<string, string> = { Kind: span.kind }
  if (span.operation) metadata.Operation = span.operation
  metadata.Origin = span.origin
  if (span.tool_type) metadata["Tool type"] = span.tool_type
  if (span.tool_call_id) metadata["Tool call ID"] = span.tool_call_id
  if (span.kind === "guardrail") {
    metadata.Profile = guardrailProfile(span) ?? "Unknown"
  }
  if (span.kind === "mcp_connect") {
    metadata.Server = mcpServer(span) ?? "Unknown"
  }
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
export function toTrace(summary: TraceSummary, detail?: TraceDetail): Trace {
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

    const byId = new Map(detail.spans.map((span) => [span.span_id, span]))
    const visit = (node: SpanNode) => {
      const { span } = node
      const verdict =
        span.kind === "guardrail"
          ? guardrailVerdict(
              span,
              span.parent_span_id ? byId.get(span.parent_span_id) : undefined,
            )
          : null
      const startMs = offset(span.start_time ?? span.end_time) ?? 0
      const endMs = offset(span.end_time) ?? startMs + (span.duration_ms ?? 0)
      observations.push({
        id: span.span_id,
        parentObservationId:
          span.parent_span_id && byId.has(span.parent_span_id)
            ? span.parent_span_id
            : (turnOf.get(span.span_id) ?? null),
        type: TYPE_BY_KIND[span.kind] ?? "SPAN",
        name: spanLabel(span),
        startMs,
        endMs: Math.max(startMs, endMs),
        level: spanLevel(span, verdict),
        statusMessage: spanStatus(span, verdict),
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
    observations,
    gateway: detail
      ? {
          events: gatewayEvents(detail),
          apiKeyId: summary.api_key_id ?? "",
          cost: summary.cost,
        }
      : undefined,
  }
}
