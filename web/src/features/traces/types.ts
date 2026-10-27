// Shapes after Langfuse's trace and observation records (see
// LANGFUSE-LICENSE.md), filled from the trace API by `fromApi.ts`.

import type { TraceGateSummary, TraceToolCost } from "@/client"

import type { ContextWeights, SessionRequests } from "./contextMap"
import type { Explanation } from "./explain"
import type { GatewayEvent, GuardrailVerdict } from "./gateway"
import type { SessionStats } from "./sessionStats"
import type { ToolCostBreakdown } from "./toolCostBreakdown"

// GUARDRAIL and MCP are the gateway's own work, which Langfuse has no type for.
export type ObservationType =
  | "SPAN"
  | "GENERATION"
  | "AGENT"
  | "TOOL"
  | "GUARDRAIL"
  | "MCP"

export type ObservationLevel = "DEFAULT" | "WARNING" | "ERROR"

export interface Observation {
  id: string
  parentObservationId: string | null
  type: ObservationType
  name: string
  /** Milliseconds from the trace's start. */
  startMs: number
  endMs: number
  level: ObservationLevel
  /** A tool call that returned an error to the agent, which read it like any other result. */
  returnedError?: boolean
  statusMessage?: string
  // GENERATION only, the fields a gateway request carries.
  model?: string
  provider?: string
  /** The whole prompt, cached tokens included. */
  promptTokens?: number
  completionTokens?: number
  cachedTokens?: number
  cacheWriteTokens?: number
  reasoningTokens?: number
  /** The model's context window, where the catalog knows it. */
  contextWindow?: number
  /** Characters per kind of input, for a request the gateway served. */
  context?: ContextWeights
  cost?: number
  input?: unknown
  output?: unknown
  metadata?: Record<string, string>
  /** A guardrail gate's check: what the gate is, what it decided and what it read. */
  gate?: GateCheck
  /** What the span means, in a sentence and a few facts. */
  explain?: Explanation
  /** Carries what the rows under it cost (a turn), so a total does not count it again. */
  isRollup?: boolean
  /** A tool call's estimated cost, read off the model requests either side of it. */
  estimate?: TraceToolCost
  /** A detail of the row above it (a permission decision, a run, a group of gates), not work of its own. */
  detail?: boolean
}

export interface GateCheck {
  gateId: string
  verdict: GuardrailVerdict
  message: string | null
  gateType: string | null
  enforcement: "required" | "advisory" | null
  runs: string[]
  origin: "repo" | "user" | null
  /** The moment this check ran at, which may be one of several in `runs`. */
  moment: string | null
  digest: string | null
  evidence: string | null
  detail: string | null
  /** What a judge gate's model calls ran on and used; null for a gate that called no model. */
  judge: JudgeCalls | null
}

export interface JudgeCalls {
  model: string
  /** The whole prompt, cached tokens included. */
  inputTokens: number
  outputTokens: number
  cacheReadTokens: number
  cacheWriteTokens: number
  /** Null until the usage ledger has priced them. */
  cost: number | null
}

export interface Trace {
  id: string
  name: string
  timestamp: string
  observations: Observation[]
  /** Absent until the session's spans have loaded. */
  gateway?: {
    events: GatewayEvent[]
    /** Each gate a coding agent's hook reported, the ones that did something first. */
    gates: TraceGateSummary[]
    /** Empty when the session's requests named no key. */
    apiKeyId: string
    cost: number
    /** Priced by Otari from what the agent reported, at the deployment's API rates, not billed through it. */
    isAtApiPrices: boolean
    /** What each tool cost, estimated; null when no call could be. */
    toolCosts: ToolCostBreakdown | null
    stats: SessionStats
    /** Its last model request with any input, and its fullest, which the context map is read from. */
    context?: SessionRequests
  }
}

export interface TraceTotals {
  latencyMs: number
  promptTokens: number
  completionTokens: number
  cachedTokens: number
  cacheWriteTokens: number
  totalTokens: number
  totalCost: number
  models: string[]
  level: ObservationLevel
}

export function summarizeTrace(trace: Trace): TraceTotals {
  let latencyMs = 0
  let promptTokens = 0
  let completionTokens = 0
  let cachedTokens = 0
  let cacheWriteTokens = 0
  let totalCost = 0
  let level: ObservationLevel = "DEFAULT"
  const models = new Set<string>()
  for (const o of trace.observations) {
    latencyMs = Math.max(latencyMs, o.endMs)
    // A turn repeats its generations' tokens as its own, so only generations count.
    if (o.type === "GENERATION") {
      promptTokens += o.promptTokens ?? 0
      completionTokens += o.completionTokens ?? 0
      cachedTokens += o.cachedTokens ?? 0
      cacheWriteTokens += o.cacheWriteTokens ?? 0
    }
    if (!o.isRollup) totalCost += o.cost ?? 0
    if (o.model) models.add(o.model)
    if (o.level === "ERROR") level = "ERROR"
    else if (o.level === "WARNING" && level !== "ERROR") level = "WARNING"
  }
  return {
    latencyMs,
    promptTokens,
    completionTokens,
    cachedTokens,
    cacheWriteTokens,
    totalTokens: promptTokens + completionTokens,
    totalCost,
    models: [...models],
    level,
  }
}
