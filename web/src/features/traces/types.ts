// Shapes after Langfuse's trace and observation records (see
// LANGFUSE-LICENSE.md), filled from the trace API by `fromApi.ts`.

import type { GatewayEvent } from "./gateway"

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
  statusMessage?: string
  // GENERATION only, the fields a gateway request carries.
  model?: string
  provider?: string
  promptTokens?: number
  completionTokens?: number
  cost?: number
  input?: unknown
  output?: unknown
  metadata?: Record<string, string>
}

export interface Trace {
  id: string
  name: string
  timestamp: string
  observations: Observation[]
  /** Absent until the session's spans have loaded. */
  gateway?: {
    events: GatewayEvent[]
    /** Empty when the session's requests named no key. */
    apiKeyId: string
    cost: number
  }
}

export interface TraceTotals {
  latencyMs: number
  promptTokens: number
  completionTokens: number
  totalTokens: number
  totalCost: number
  models: string[]
  level: ObservationLevel
}

export function summarizeTrace(trace: Trace): TraceTotals {
  let latencyMs = 0
  let promptTokens = 0
  let completionTokens = 0
  let totalCost = 0
  let level: ObservationLevel = "DEFAULT"
  const models = new Set<string>()
  for (const o of trace.observations) {
    latencyMs = Math.max(latencyMs, o.endMs)
    // A turn repeats its generations' tokens as its own, so only generations count.
    if (o.type === "GENERATION") {
      promptTokens += o.promptTokens ?? 0
      completionTokens += o.completionTokens ?? 0
    }
    totalCost += o.cost ?? 0
    if (o.model) models.add(o.model)
    if (o.level === "ERROR") level = "ERROR"
    else if (o.level === "WARNING" && level !== "ERROR") level = "WARNING"
  }
  return {
    latencyMs,
    promptTokens,
    completionTokens,
    totalTokens: promptTokens + completionTokens,
    totalCost,
    models: [...models],
    level,
  }
}
