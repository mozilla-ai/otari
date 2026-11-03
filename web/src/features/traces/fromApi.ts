import type {
  TraceDetail,
  TraceGateSummary,
  TraceSpan,
  TraceSummary,
  TraceTurn,
} from "@/client"
import { contextWeightsOf, sessionRequests } from "./contextMap"
import {
  explainSpan,
  type GateTally,
  gateGroupExplanation,
  groupWords,
} from "./explain"
import {
  failureLabel,
  type GuardrailVerdict,
  gateVerdict,
  gatewayEvents,
  guardrailProfile,
  guardrailVerdict,
  mcpServer,
} from "./gateway"
import { inSubagentOf, sessionStats } from "./sessionStats"
import { toolCostBreakdown } from "./toolCostBreakdown"
import {
  buildTraceTree,
  type SpanNode,
  sessionName,
  spanLabel,
} from "./traceModel"
import type {
  GateCheck,
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

// A gate that fired is the guardrail working, so it asks for attention, not alarm.
// Only one that could not decide is a failure.
const GATE_LEVEL: Record<GuardrailVerdict, ObservationLevel> = {
  passed: "DEFAULT",
  blocked: "WARNING",
  flagged: "WARNING",
  refused: "ERROR",
  unchecked: "ERROR",
}

const GATE_STATUS: Record<GuardrailVerdict, string | undefined> = {
  passed: undefined,
  blocked: "Blocked the agent's action.",
  flagged: "Flagged. The gate is advisory, so the agent went ahead.",
  refused: "Could not decide, and blocked the agent's action.",
  unchecked: "Could not decide. The action went ahead unchecked.",
}

const REFUSED_BY: Record<string, string> = {
  hook: "Blocked by a hook before it ran.",
  user: "Refused by the user.",
  config: "Refused by a permission rule.",
}

function spanLevel(
  span: TraceSpan,
  verdict: GuardrailVerdict | null,
): ObservationLevel {
  if (verdict) {
    return gateVerdict(span) ? GATE_LEVEL[verdict] : GUARDRAIL_LEVEL[verdict]
  }
  if (span.outcome === "blocked") return "WARNING"
  if (span.outcome !== "error") return "DEFAULT"
  return span.recovered ? "WARNING" : "ERROR"
}

function returnedError(span: TraceSpan): boolean {
  return (
    span.kind === "tool" && span.attributes["otari.tool.succeeded"] === false
  )
}

function spanStatus(
  span: TraceSpan,
  verdict: GuardrailVerdict | null,
  gateMessages: ReadonlyMap<string, string>,
  blockedBy: readonly string[] = [],
): string | undefined {
  if (verdict && gateVerdict(span)) {
    const said = GATE_STATUS[verdict]
    const gateId = span.attributes["otari.gate.id"]
    const message =
      typeof gateId === "string" ? gateMessages.get(gateId) : undefined
    // A passing gate's message describes a failure it did not find.
    return said && message ? `${said} ${message}` : said
  }
  if (verdict) return GUARDRAIL_STATUS[verdict]
  if (span.outcome === "blocked" && blockedBy.length > 0) {
    const said = blockedBy
      .map((gateId) => gateMessages.get(gateId))
      .filter((message) => message !== undefined)
    return [`Blocked by ${blockedBy.join(", ")} before it ran.`, ...said].join(
      " ",
    )
  }
  if (span.outcome === "blocked") {
    const source = span.attributes["otari.tool.decision_source"]
    return (
      (typeof source === "string" &&
        REFUSED_BY[source.startsWith("user_") ? "user" : source]) ||
      "Blocked before it ran."
    )
  }
  if (returnedError(span)) {
    return "Returned an error, which the agent read like any other result. A command returns one when it exits with a non-zero status, as grep finding nothing or a failing test does, so it is not counted as an error of the session."
  }
  if (span.outcome !== "error") return undefined
  if (span.kind === "mcp_connect") {
    return "Connecting to the server or listing its tools failed."
  }
  const error = failureLabel(span.error_class)
  return span.recovered ? `${error}, recovered by a fallback` : error
}

// What the span was about, where the deployment keeps content: the one value on
// its own, so a command or a prompt reads as text rather than as an object. A
// span whose content the reader may not open says so rather than showing none.
export const PRIVATE_CONTENT =
  "Private to the person whose session this is. Anyone else reads it only through their share or a recorded break-glass opening."

function spanInput(span: TraceSpan): unknown {
  if (span.content_private) return PRIVATE_CONTENT
  const values = Object.values(span.content)
  if (values.length === 0) return undefined
  return values.length === 1 ? values[0] : span.content
}

function stringAttribute(span: TraceSpan, key: string): string | null {
  const value = span.attributes[key]
  return typeof value === "string" ? value : null
}

function gateOrigin(value: string | null): GateCheck["origin"] {
  return value === "repo" || value === "user" ? value : null
}

// A gate's check, joined to what the session's guardrail says about the gate.
// What the check read travels on one of its gate spans, found by the OTLP trace
// every gate of one check shares.
function gateCheck(
  span: TraceSpan,
  gates: ReadonlyMap<string, TraceGateSummary>,
  checkEvidence: ReadonlyMap<string, string>,
): GateCheck | undefined {
  const verdict = gateVerdict(span)
  const gateId = stringAttribute(span, "otari.gate.id")
  if (!verdict || !gateId) return undefined
  const declared = gates.get(gateId)
  return {
    gateId,
    verdict,
    message: declared?.message ?? null,
    gateType: declared?.gate_type ?? stringAttribute(span, "otari.gate.type"),
    enforcement: declared?.enforcement ?? null,
    runs: declared?.runs ?? [],
    origin:
      declared?.origin ??
      gateOrigin(stringAttribute(span, "otari.gate.origin")),
    moment: span.operation,
    digest: stringAttribute(span, "otari.policy.digest"),
    evidence:
      span.content["gate.evidence"] ??
      (span.otel_trace_id ? checkEvidence.get(span.otel_trace_id) : null) ??
      null,
    detail: span.content["gate.detail"] ?? null,
    judge: span.model
      ? {
          model: span.model,
          inputTokens: span.input_tokens ?? 0,
          outputTokens: span.output_tokens ?? 0,
          cacheReadTokens: span.cache_read_tokens ?? 0,
          cacheWriteTokens: span.cache_write_tokens ?? 0,
          cost: span.cost ?? null,
        }
      : null,
  }
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
          cachedTokens: turn.cache_read_tokens,
          cost: turn.cost > 0 ? turn.cost : undefined,
          isRollup: true,
          metadata: {
            State: turn.state,
            "LLM calls": String(turn.llm_calls),
            "Tool calls": String(turn.tool_calls),
          },
        })
      }
    }

    const byId = new Map(detail.spans.map((span) => [span.span_id, span]))
    const estimates = new Map(
      detail.tool_costs.map((cost) => [cost.span_id, cost]),
    )
    // The gates that stopped each tool call, by the tool call they hang under.
    const blockedBy = new Map<string, string[]>()
    for (const span of detail.spans) {
      const gateId = span.attributes["otari.gate.id"]
      const verdict = gateVerdict(span)
      if (
        (verdict === "blocked" || verdict === "refused") &&
        span.parent_span_id &&
        typeof gateId === "string"
      ) {
        blockedBy.set(span.parent_span_id, [
          ...(blockedBy.get(span.parent_span_id) ?? []),
          gateId,
        ])
      }
    }
    const gateSummaries = new Map(
      detail.gates.map((gate) => [gate.gate_id, gate] as const),
    )
    const checkEvidence = new Map(
      detail.spans.flatMap((span) => {
        const evidence = span.content["gate.evidence"]
        return evidence && span.otel_trace_id
          ? [[span.otel_trace_id, evidence] as const]
          : []
      }),
    )
    const gateMessages = new Map(
      detail.gates.flatMap((gate) =>
        gate.message ? [[gate.gate_id, gate.message] as const] : [],
      ),
    )
    const isGate = (node: SpanNode) =>
      node.span.kind === "guardrail" &&
      typeof node.span.attributes["otari.gate.id"] === "string"
    const tallyOf = (nodes: SpanNode[]): GateTally => {
      const tally: GateTally = {}
      for (const { span } of nodes) {
        const verdict = gateVerdict(span)
        if (verdict) tally[verdict] = (tally[verdict] ?? 0) + 1
      }
      return tally
    }
    const parentOf = (span: TraceSpan) =>
      span.parent_span_id && byId.has(span.parent_span_id)
        ? span.parent_span_id
        : (turnOf.get(span.span_id) ?? null)
    // A gateway request that made one LLM call and nothing else is that call,
    // so the call takes the request's row rather than sitting alone under it.
    // A request with a fallback, a gateway-run tool, a gate, or content or an
    // error of its own keeps its row, so nothing it carries is lost.
    const loneCall = ({ span, children }: SpanNode) => {
      const [call] = children
      return span.kind === "step" &&
        span.operation !== "interaction" &&
        Object.keys(span.content).length === 0 &&
        !span.content_private &&
        span.error_class === null &&
        children.length === 1 &&
        call.span.kind === "llm" &&
        call.children.length === 0 &&
        call.span.outcome === span.outcome
        ? call
        : undefined
    }
    const adoptedBy = new Map<string, string | null>()
    // A span's gates go under one row of their own, so a tool call reads as one
    // line until it is opened, and its gates as one more until they are.
    const visit = (node: SpanNode, parentOverride?: string) => {
      const { span } = node
      const call = loneCall(node)
      if (call) {
        adoptedBy.set(call.span.span_id, parentOf(span))
        visit(call)
        return
      }
      const adopter = adoptedBy.get(span.span_id)
      const gateNodes = node.children.filter(isGate)
      const tally = gateNodes.length ? tallyOf(gateNodes) : undefined
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
          parentOverride ?? (adopter === undefined ? parentOf(span) : adopter),
        explain: explainSpan(
          span,
          span.parent_span_id ? byId.get(span.parent_span_id) : undefined,
          blockedBy.get(span.span_id) ?? [],
          tally,
        ),
        type: TYPE_BY_KIND[span.kind] ?? "SPAN",
        name: spanLabel(span),
        input: spanInput(span),
        gate: gateCheck(span, gateSummaries, checkEvidence),
        detail:
          span.operation === "tool.blocked_on_user" ||
          span.operation === "tool.execution" ||
          parentOverride !== undefined,
        startMs,
        endMs: Math.max(startMs, endMs),
        level: spanLevel(span, verdict),
        returnedError: returnedError(span),
        statusMessage: spanStatus(
          span,
          verdict,
          gateMessages,
          blockedBy.get(span.span_id),
        ),
        model: span.model ?? undefined,
        provider: span.provider ?? undefined,
        promptTokens: span.input_tokens ?? undefined,
        completionTokens: span.output_tokens ?? undefined,
        cachedTokens: span.cache_read_tokens ?? undefined,
        cacheWriteTokens: span.cache_write_tokens ?? undefined,
        reasoningTokens: span.reasoning_tokens ?? undefined,
        contextWindow: span.context_window ?? undefined,
        context: contextWeightsOf(span.attributes),
        cost: span.cost ?? undefined,
        requestGroupId:
          span.kind === "llm" && span.origin === "gateway"
            ? (span.request_id ?? undefined)
            : undefined,
        estimate: estimates.get(span.span_id),
        metadata: spanMetadata(span),
      })
      for (const child of node.children) if (!isGate(child)) visit(child)
      if (tally) {
        const groupId = `${span.span_id}:gates`
        const starts = gateNodes.map(
          (gate) => offset(gate.span.start_time ?? gate.span.end_time) ?? 0,
        )
        const ends = gateNodes.map(
          (gate) => offset(gate.span.end_time) ?? startMs,
        )
        observations.push({
          id: groupId,
          parentObservationId: span.span_id,
          type: "GUARDRAIL",
          detail: true,
          name: `Gates · ${groupWords(tally, gateNodes.length)}`,
          startMs: Math.min(...starts),
          endMs: Math.max(...starts, ...ends),
          level:
            tally.refused || tally.unchecked
              ? "ERROR"
              : tally.blocked || tally.flagged
                ? "WARNING"
                : "DEFAULT",
          explain: gateGroupExplanation(
            tally,
            span.kind === "step",
            gateNodes.length,
          ),
        })
        for (const gate of gateNodes) visit(gate, groupId)
      }
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
          gates: detail.gates,
          apiKeyId: summary.api_key_id ?? "",
          cost: summary.cost,
          isAtApiPrices: detail.spans.some(
            (span) =>
              span.kind === "llm" &&
              span.origin === "otlp" &&
              span.cost != null,
          ),
          toolCosts: toolCostBreakdown(detail, summary.cost),
          stats: sessionStats(summary, detail),
          context: sessionRequests(observations, inSubagentOf(detail.spans)),
        }
      : undefined,
  }
}
