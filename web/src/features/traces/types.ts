// Shapes modeled on Langfuse's trace and observation records
// (MIT, Copyright (c) 2023-2026 ClickHouse, Inc.), carrying Otari-shaped
// fields on generations. Mock data only on this branch.

export type ObservationType = "SPAN" | "GENERATION" | "AGENT" | "TOOL" | "EVENT"

export type ObservationLevel = "DEBUG" | "DEFAULT" | "WARNING" | "ERROR"

export interface Observation {
  id: string
  traceId: string
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
  inputCost?: number
  outputCost?: number
  input?: unknown
  output?: unknown
  metadata?: Record<string, string>
}

export interface Trace {
  id: string
  name: string
  timestamp: string
  userId: string
  sessionId: string
  environment: string
  tags: string[]
  input: unknown
  output: unknown
  metadata: Record<string, string>
  observations: Observation[]
}

export interface TraceSummary {
  latencyMs: number
  promptTokens: number
  completionTokens: number
  totalTokens: number
  totalCost: number
  inputCost: number
  outputCost: number
  errorCount: number
  warningCount: number
  observationCount: number
  models: string[]
  level: ObservationLevel
}

export function summarizeTrace(trace: Trace): TraceSummary {
  let latencyMs = 0
  let promptTokens = 0
  let completionTokens = 0
  let inputCost = 0
  let outputCost = 0
  let errorCount = 0
  let warningCount = 0
  let level: ObservationLevel = "DEFAULT"
  const models = new Set<string>()
  for (const o of trace.observations) {
    latencyMs = Math.max(latencyMs, o.endMs)
    promptTokens += o.promptTokens ?? 0
    completionTokens += o.completionTokens ?? 0
    inputCost += o.inputCost ?? 0
    outputCost += o.outputCost ?? 0
    if (o.level === "ERROR") errorCount += 1
    if (o.level === "WARNING") warningCount += 1
    if (o.model) models.add(o.model)
    if (o.level === "ERROR") level = "ERROR"
    else if (o.level === "WARNING" && level !== "ERROR") level = "WARNING"
  }
  return {
    latencyMs,
    promptTokens,
    completionTokens,
    totalTokens: promptTokens + completionTokens,
    totalCost: inputCost + outputCost,
    inputCost,
    outputCost,
    errorCount,
    warningCount,
    observationCount: trace.observations.length,
    models: [...models],
    level,
  }
}
