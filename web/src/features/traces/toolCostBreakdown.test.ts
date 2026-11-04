import { describe, expect, it } from "vitest"

import type { TraceToolCost } from "@/client"
import { traceDetail, traceSpan } from "@/tests/traceFixtures"

import { toolCostBreakdown } from "./toolCostBreakdown"

function estimate(
  span_id: string,
  total: number,
  subagent: number | null = null,
): TraceToolCost {
  return {
    span_id,
    calling: 0,
    writing: 0,
    rereading: total - (subagent ?? 0),
    result_tokens: 0,
    later_requests: 0,
    calls_sharing_request: 1,
    calls_sharing_result: 1,
    subagent,
    total,
  }
}

const tool = (span_id: string, tool_name: string, parent = "turn") =>
  traceSpan({ span_id, parent_span_id: parent, kind: "tool", tool_name })

// A turn that ran a Bash call itself and an Agent call whose subagent, on
// Haiku, ran two Bash calls and a Read.
const spans = [
  traceSpan({ span_id: "turn", kind: "step", opens_turn: true }),
  tool("bash", "Bash"),
  tool("agent", "Agent"),
  traceSpan({
    span_id: "run",
    parent_span_id: "agent",
    kind: "other",
    operation: "tool.execution",
  }),
  traceSpan({
    span_id: "sub-llm",
    parent_span_id: "run",
    kind: "llm",
    model: "claude-haiku-4-5",
  }),
  tool("sub-bash-1", "Bash", "run"),
  tool("sub-bash-2", "Bash", "run"),
  tool("sub-read", "Read", "run"),
]

describe("toolCostBreakdown", () => {
  it("lists a subagent's calls under the call that ran it, counted once", () => {
    const breakdown = toolCostBreakdown(
      traceDetail({
        spans,
        tool_costs: [
          estimate("bash", 0.1),
          estimate("agent", 2.5, 2.4),
          estimate("sub-bash-1", 0.6),
          estimate("sub-bash-2", 0.9),
          estimate("sub-read", 0.3),
        ],
      }),
      3,
    )

    expect(breakdown).not.toBeNull()
    const [agent, bash] = breakdown?.rows ?? []
    expect(agent).toMatchObject({
      tool: "Agent",
      calls: 1,
      cost: 2.5,
      models: ["claude-haiku-4-5"],
    })
    expect(agent.ownRequests).toBeCloseTo(0.6)
    expect(
      agent.inside.map((row) => [row.tool, row.calls, row.cost, row.spanId]),
    ).toEqual([
      ["Bash", 2, 1.5, "sub-bash-2"],
      ["Read", 1, 0.3, "sub-read"],
    ])
    expect(bash).toMatchObject({ tool: "Bash", calls: 1, cost: 0.1 })
    // The prompts and answers: what the session cost beyond its tool calls.
    expect(breakdown?.untied).toBeCloseTo(0.4)
  })

  it("keeps what judge gates cost out of what no tool call accounts for", () => {
    const breakdown = toolCostBreakdown(
      traceDetail({
        spans: [
          ...spans,
          traceSpan({
            span_id: "judge",
            parent_span_id: "turn",
            kind: "guardrail",
            model: "claude-haiku-4-5-20251001",
            cost: 0.15,
          }),
        ],
        tool_costs: [estimate("bash", 0.1)],
      }),
      1,
    )

    expect(breakdown?.judges).toBeCloseTo(0.15)
    expect(breakdown?.untied).toBeCloseTo(0.75)
  })

  it("reads a session whose spans name each other as parents", () => {
    const looped = [
      tool("a", "Bash", "b"),
      tool("b", "Read", "a"),
      traceSpan({
        span_id: "run",
        parent_span_id: "a",
        kind: "other",
        operation: "tool.execution",
      }),
    ]

    const breakdown = toolCostBreakdown(
      traceDetail({ spans: looped, tool_costs: [estimate("a", 0.1)] }),
      0.1,
    )

    expect(breakdown?.rows.map((row) => row.tool)).toEqual(["Bash"])
  })

  it("has nothing to say about a session with no estimates", () => {
    expect(toolCostBreakdown(traceDetail({ tool_costs: [] }), 1)).toBeNull()
  })
})
