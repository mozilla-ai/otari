import type { TraceDetail, TraceSpan } from "@/client"
import type { ChipTone } from "@/design-system/indicators/Chip"

// What the gateway decided in a session, as opposed to what the agent did: its
// guardrail verdicts, the fallbacks and model switches routing made, and the
// MCP servers it could not reach. Only spans the gateway recorded say any of
// this, so an instrumented agent's own spans never produce an event.

export type GuardrailVerdict =
  | "passed"
  | "blocked"
  | "flagged"
  | "refused"
  | "unchecked"

// A flag stops the request only in block mode, and a check that could not run
// stops it only when the guardrail fails closed. Neither setting is recorded on
// the span, so the step it ran under says which happened.
export function guardrailVerdict(
  span: TraceSpan,
  step: TraceSpan | undefined,
): GuardrailVerdict {
  if (span.outcome !== "error") return "passed"
  const stopped = step?.outcome === "error"
  if (span.error_class === "guardrail_unavailable") {
    return stopped ? "refused" : "unchecked"
  }
  const mode = span.attributes["otari.guardrail.mode"]
  return mode === "monitor" || (mode === undefined && !stopped)
    ? "flagged"
    : "blocked"
}

export function guardrailProfile(span: TraceSpan): string | null {
  const profile = span.attributes["otari.guardrail.profile"]
  return typeof profile === "string" ? profile : null
}

export function mcpServer(span: TraceSpan): string | null {
  const server = span.attributes["otari.mcp.server"]
  return typeof server === "string" ? server : null
}

export interface GatewayEvent {
  /** The span the event is about, selected in the tree when the row is picked. */
  spanId: string
  at: string | null
  word: string
  tone: ChipTone
  title: string
  detail: string
}

function at(span: TraceSpan): string | null {
  return span.start_time ?? span.end_time
}

function byStart(a: TraceSpan, b: TraceSpan): number {
  return (at(a) ?? "").localeCompare(at(b) ?? "")
}

export function failureLabel(errorClass: string | null): string {
  if (!errorClass) return "error"
  const status = /^status_(\d{3})$/.exec(errorClass)
  if (status) return `HTTP ${status[1]}`
  return errorClass.replaceAll("_", " ")
}

function modelName(span: TraceSpan): string {
  return span.model ?? "the model"
}

function capitalize(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1)
}

function guardrailEvent(
  span: TraceSpan,
  step: TraceSpan | undefined,
): GatewayEvent | null {
  const profile = guardrailProfile(span)
  const name = profile ? `the ${profile} guardrail` : "a guardrail"
  const base = { spanId: span.span_id, at: at(span) }
  switch (guardrailVerdict(span, step)) {
    case "passed":
      return null
    case "blocked":
      return {
        ...base,
        word: "Blocked",
        tone: "danger",
        title: `Blocked by ${name}`,
        detail: "Refused before it reached a model.",
      }
    case "flagged":
      return {
        ...base,
        word: "Flagged",
        tone: "warning",
        title: `Flagged by ${name}`,
        detail: "Monitor mode, so the request still went through.",
      }
    case "refused":
      return {
        ...base,
        word: "Unavailable",
        tone: "danger",
        title: `${capitalize(name)} could not run`,
        detail: "It fails closed, so the request was refused.",
      }
    case "unchecked":
      return {
        ...base,
        word: "Unavailable",
        tone: "warning",
        title: `${capitalize(name)} could not run`,
        detail: "The request went through unchecked.",
      }
  }
}

export function gatewayEvents(detail: TraceDetail): GatewayEvent[] {
  const spans = detail.spans.filter((span) => span.origin === "gateway")
  const byId = new Map(spans.map((span) => [span.span_id, span]))
  const calls = spans.filter((span) => span.kind === "llm").sort(byStart)
  const events: GatewayEvent[] = []

  for (const span of spans) {
    if (span.kind === "guardrail") {
      const event = guardrailEvent(
        span,
        span.parent_span_id ? byId.get(span.parent_span_id) : undefined,
      )
      if (event) events.push(event)
    }
    if (span.kind === "mcp_connect" && span.outcome === "error") {
      events.push({
        spanId: span.span_id,
        at: at(span),
        word: "Unreachable",
        tone: "danger",
        title: `Could not reach ${mcpServer(span) ? `the ${mcpServer(span)} MCP server` : "an MCP server"}`,
        detail: "Connecting or listing its tools failed.",
      })
    }
  }

  const fellBack = new Set<string>()
  for (const span of calls) {
    if (!span.recovered) continue
    const answer = calls.find(
      (other) =>
        other.request_id === span.request_id &&
        other.outcome === "ok" &&
        byStart(other, span) > 0,
    )
    if (span.request_id) fellBack.add(span.request_id)
    events.push({
      spanId: span.span_id,
      at: at(span),
      word: "Fallback",
      tone: "warning",
      title: answer
        ? `${modelName(span)} failed, ${modelName(answer)} answered`
        : `${modelName(span)} failed, a later attempt answered`,
      detail: `Failed with ${failureLabel(span.error_class)}. Routing tried the next candidate.`,
    })
  }

  // A switch is read between the calls that answered, so a fallback's failed
  // attempt never counts as the model the session was on.
  let previous: TraceSpan | null = null
  for (const span of calls) {
    if (span.outcome !== "ok" || !span.model) continue
    if (
      previous?.model &&
      (previous.model !== span.model || previous.provider !== span.provider)
    ) {
      const sameModel = previous.model === span.model
      const policy = span.attributes["otari.routing.policy"]
      const reason = span.attributes["otari.routing.selection_reason"]
      const why =
        span.request_id && fellBack.has(span.request_id)
          ? "A fallback moved the session. "
          : typeof policy === "string"
            ? `Routing policy ${policy} chose it${typeof reason === "string" ? ` (${reason.replaceAll("_", " ")})` : ""}. `
            : ""
      events.push({
        spanId: span.span_id,
        at: at(span),
        word: "Switched",
        tone: "info",
        title: sameModel
          ? `${span.model} moved from ${previous.provider ?? "one provider"} to ${span.provider ?? "another"}`
          : `Switched from ${previous.model} to ${span.model}`,
        detail: `${why}The prompt cache built on ${previous.model}${sameModel && previous.provider ? ` at ${previous.provider}` : ""} does not carry over.`,
      })
    }
    previous = span
  }

  return events.sort((a, b) => (a.at ?? "").localeCompare(b.at ?? ""))
}
