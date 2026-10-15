import { describe, expect, it } from "vitest"

import { traceDetail, traceSpan, traceSummary } from "@/tests/traceFixtures"

import { toTrace } from "./fromApi"
import { summarizeTrace } from "./types"

describe("toTrace", () => {
  it("draws a session's spans as observations under the steps they belong to", () => {
    const trace = toTrace(traceSummary(), traceDetail())
    const byId = new Map(trace.observations.map((o) => [o.id, o]))

    // One turn is not wrapped in a turn of its own.
    expect(trace.observations.some((o) => o.id.startsWith("turn-"))).toBe(false)
    expect(byId.get("req-1")).toMatchObject({
      type: "SPAN",
      name: "Request",
      parentObservationId: null,
      startMs: 0,
      endMs: 10_000,
    })
    expect(byId.get("llm-1")).toMatchObject({
      type: "GENERATION",
      name: "gpt-5",
      parentObservationId: "req-1",
      promptTokens: 10,
      completionTokens: 2,
      cost: 0.01,
      level: "DEFAULT",
    })
    expect(byId.get("call-toolu_1")).toMatchObject({
      type: "TOOL",
      name: "Bash",
      parentObservationId: "req-1",
      startMs: 10_000,
      endMs: 60_000,
    })
    expect(byId.get("call-toolu_1")?.metadata?.Timing).toBe("Approximate")
  })

  it("marks a failure a fallback recovered as a warning, not an error", () => {
    const trace = toTrace(traceSummary(), traceDetail())
    const recovered = trace.observations.find((o) => o.id === "llm-2")

    expect(recovered?.level).toBe("WARNING")
    expect(recovered?.statusMessage).toBe("error, recovered by a fallback")
    expect(summarizeTrace(trace)).toMatchObject({
      level: "WARNING",
      totalCost: 0.01,
    })
  })

  it("draws guardrail checks and MCP connections as their own types", () => {
    const trace = toTrace(
      traceSummary({ api_key_id: "key-1" }),
      traceDetail({
        spans: [
          traceSpan({ span_id: "req-1" }),
          traceSpan({
            span_id: "g-1",
            parent_span_id: "req-1",
            kind: "guardrail",
            outcome: "error",
            error_class: "guardrail_flagged",
            attributes: {
              "otari.guardrail.profile": "pii",
              "otari.guardrail.mode": "monitor",
            },
          }),
          traceSpan({
            span_id: "mcp-1",
            parent_span_id: "req-1",
            kind: "mcp_connect",
            outcome: "error",
            error_class: "mcp_unreachable",
            attributes: { "otari.mcp.server": "github" },
          }),
        ],
      }),
    )
    const byId = new Map(trace.observations.map((o) => [o.id, o]))

    // A monitor-mode flag let the request through, so it is not an error.
    expect(byId.get("g-1")).toMatchObject({
      type: "GUARDRAIL",
      name: "pii",
      level: "WARNING",
    })
    expect(byId.get("mcp-1")).toMatchObject({
      type: "MCP",
      name: "github",
      level: "ERROR",
    })
    expect(trace.gateway).toMatchObject({ apiKeyId: "key-1", cost: 0.02 })
    expect(trace.gateway?.events.map((e) => e.word)).toEqual([
      "Flagged",
      "Unreachable",
    ])
  })

  it("has no gateway report until the spans load", () => {
    expect(toTrace(traceSummary()).gateway).toBeUndefined()
  })

  it("puts each turn's steps under the turn once a session has several", () => {
    const base = traceDetail()
    const detail = traceDetail({
      turns: [
        { ...base.turns[0], step_ids: ["req-1"] },
        {
          ...base.turns[0],
          index: 1,
          state: "failed",
          started_at: "2026-10-07T12:01:00Z",
          step_ids: ["req-2"],
        },
      ],
    })

    const trace = toTrace(traceSummary(), detail)
    const byId = new Map(trace.observations.map((o) => [o.id, o]))

    expect(byId.get("turn-0")).toMatchObject({
      name: "Turn 1",
      level: "DEFAULT",
    })
    expect(byId.get("turn-1")).toMatchObject({
      name: "Turn 2",
      level: "ERROR",
      startMs: 60_000,
    })
    expect(byId.get("req-1")?.parentObservationId).toBe("turn-0")
    expect(byId.get("req-2")?.parentObservationId).toBe("turn-1")
    expect(byId.get("llm-2")?.parentObservationId).toBe("req-2")
    // The turns carry their steps' tokens too, and are not counted twice.
    expect(summarizeTrace(trace).totalTokens).toBe(12)
  })

  it("keeps an instrumented agent's own span tree", () => {
    const detail = traceDetail({
      turns: [],
      spans: [
        traceSpan({ span_id: "a1", kind: "agent", name: "invoke_agent" }),
        traceSpan({
          span_id: "x1",
          parent_span_id: "a1",
          kind: "tool",
          tool_name: "grep",
          outcome: "error",
          error_class: "timeout",
        }),
      ],
    })

    const [agent, tool] = toTrace(traceSummary(), detail).observations

    expect(agent).toMatchObject({ type: "AGENT", parentObservationId: null })
    expect(tool).toMatchObject({
      type: "TOOL",
      parentObservationId: "a1",
      level: "ERROR",
      statusMessage: "timeout",
    })
  })

  it("is the list row alone, with no observations, before the spans arrive", () => {
    const trace = toTrace(traceSummary({ session_source: "none" }))

    expect(trace).toMatchObject({
      id: traceSummary().trace_id,
      name: "claude-code",
      observations: [],
      gateway: undefined,
    })
  })
})
