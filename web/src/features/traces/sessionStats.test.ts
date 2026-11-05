import { describe, expect, it } from "vitest"

import { traceDetail, traceSpan, traceSummary } from "@/tests/traceFixtures"

import { sessionStats } from "./sessionStats"

const T0 = Date.parse("2026-10-19T09:00:00Z")
const at = (seconds: number) => new Date(T0 + seconds * 1000).toISOString()

const RATES = {
  "otari.llm.rate.input": "3",
  "otari.llm.rate.output": "15",
  "otari.llm.rate.cache_read": "0.3",
  "otari.llm.rate.cache_write": "3.75",
}

// One turn over 100 s: two model requests, a Bash call that waited 20 s for
// approval and ran 5 s, a hook that took 4 s, a gateway guardrail on a request,
// and an Agent call whose subagent ran on long after the call's own run closed.
const spans = [
  traceSpan({
    span_id: "turn",
    kind: "step",
    opens_turn: true,
    start_time: at(0),
    end_time: at(100),
  }),
  traceSpan({
    span_id: "r1",
    parent_span_id: "turn",
    kind: "llm",
    start_time: at(0),
    end_time: at(10),
    input_tokens: 11_000,
    output_tokens: 200,
    cache_read_tokens: 0,
    cache_write_tokens: 10_000,
    attributes: RATES,
  }),
  traceSpan({
    span_id: "bash",
    parent_span_id: "turn",
    kind: "tool",
    tool_name: "Bash",
    start_time: at(10),
    end_time: at(35),
  }),
  traceSpan({
    span_id: "bash-wait",
    parent_span_id: "bash",
    kind: "other",
    operation: "tool.blocked_on_user",
    start_time: at(10),
    end_time: at(30),
  }),
  traceSpan({
    span_id: "bash-run",
    parent_span_id: "bash",
    kind: "other",
    operation: "tool.execution",
    start_time: at(30),
    end_time: at(35),
  }),
  traceSpan({
    span_id: "gate",
    parent_span_id: "bash",
    kind: "guardrail",
    start_time: at(10),
    end_time: at(14),
    attributes: { "otari.gate.outcome": "pass" },
  }),
  traceSpan({
    span_id: "gate-2",
    parent_span_id: "bash",
    kind: "guardrail",
    start_time: at(10),
    end_time: at(11),
    attributes: { "otari.gate.outcome": "fail", "otari.gate.blocked": true },
  }),
  traceSpan({
    span_id: "gateway-guardrail",
    parent_span_id: "r1",
    kind: "guardrail",
    start_time: at(0),
    end_time: at(2),
  }),
  traceSpan({
    span_id: "agent",
    parent_span_id: "turn",
    kind: "tool",
    tool_name: "Agent",
    start_time: at(40),
    end_time: at(41),
  }),
  traceSpan({
    span_id: "agent-run",
    parent_span_id: "agent",
    kind: "other",
    operation: "tool.execution",
    start_time: at(40),
    end_time: at(41),
  }),
  traceSpan({
    span_id: "sub-r1",
    parent_span_id: "agent-run",
    kind: "llm",
    start_time: at(41),
    end_time: at(90),
    input_tokens: 10_000,
    output_tokens: 100,
    cache_read_tokens: 10_000,
    cache_write_tokens: 0,
    attributes: RATES,
  }),
  traceSpan({
    span_id: "handback",
    parent_span_id: "agent-run",
    kind: "tool",
    tool_name: "SubagentHandback",
    start_time: at(90),
    end_time: at(90),
  }),
]

describe("sessionStats", () => {
  const stats = sessionStats(
    traceSummary({ started_at: at(0), last_activity_at: at(100) }),
    traceDetail({ spans }),
  )

  it("counts the session's turns, requests and tool calls, but not a subagent's handback", () => {
    expect(stats).toMatchObject({
      lengthMs: 100_000,
      requests: 2,
      toolCalls: 2,
      inSubagents: 0,
    })
  })

  it("counts the gates the hooks ran, by what each decided", () => {
    expect(stats.gates).toEqual({
      runs: 2,
      verdicts: { passed: 1, blocked: 1 },
    })
  })

  it("says where the time went, apart from waiting for approval, and counts only gates as hooks", () => {
    // Bash ran 5 s, and the Agent call ran until its subagent's last span, 50 s.
    expect(stats.time).toEqual({
      modelMs: 59_000,
      toolsMs: 55_000,
      waitingMs: 20_000,
      hooksMs: 4_000,
    })
  })

  it("counts the time something was happening once, so the rest of the session reads as idle", () => {
    // 0-35 s (requests, the wait, the hooks, Bash's run) and 40-90 s (the
    // Agent's run and its subagent's request); 35-40 s and 90-100 s were idle.
    expect(stats.activeMs).toBe(85_000)
  })

  it("reads a session as at least as long as what happened in it", () => {
    // The summary's clock stopped before the request it recorded ended.
    const late = sessionStats(
      traceSummary({ started_at: at(0), last_activity_at: at(0) }),
      traceDetail({
        spans: [
          traceSpan({
            span_id: "r",
            kind: "llm",
            start_time: at(0),
            end_time: at(8),
          }),
        ],
      }),
    )

    expect(late.lengthMs).toBe(8_000)
  })

  it("finds no tool time inside a request when tools ran between them", () => {
    expect(stats.toolsInModelMs).toBe(0)
  })

  it("runs an Agent call until its subagent's last span, not its own run's", () => {
    expect(stats.tools).toEqual([
      { tool: "Agent", calls: 1, runningMs: 50_000 },
      { tool: "Bash", calls: 1, runningMs: 5_000 },
    ])
  })

  it("prices each meter and says what reading the cache saved", () => {
    expect(stats.cost?.input).toBeCloseTo(0.003)
    expect(stats.cost?.output).toBeCloseTo(0.0045)
    expect(stats.cost?.cacheRead).toBeCloseTo(0.003)
    expect(stats.cost?.cacheWrite).toBeCloseTo(0.0375)
    expect(stats.cacheSaved).toBeCloseTo(0.027)
    expect(stats.pricedRequests).toBe(2)
  })

  it("counts the judge gates that called a model, and what the priced ones cost", () => {
    const judged = sessionStats(
      traceSummary(),
      traceDetail({
        spans: [
          traceSpan({
            span_id: "judge-1",
            kind: "guardrail",
            model: "claude-haiku-4-5-20251001",
            cost: 0.02,
            attributes: { "otari.gate.outcome": "pass" },
          }),
          traceSpan({
            span_id: "judge-2",
            kind: "guardrail",
            model: "claude-haiku-4-5-20251001",
            attributes: { "otari.gate.outcome": "fail" },
          }),
          traceSpan({
            span_id: "path",
            kind: "guardrail",
            attributes: { "otari.gate.outcome": "pass" },
          }),
        ],
      }),
    )

    expect(judged.gates.runs).toBe(3)
    expect(judged.judges).toEqual({
      calls: 2,
      cost: 0.02,
      priced: 1,
    })
  })

  it("has no cost by meter for a session whose requests were not priced by the token", () => {
    const unpriced = sessionStats(
      traceSummary(),
      traceDetail({
        spans: [traceSpan({ span_id: "r", kind: "llm", input_tokens: 5 })],
      }),
    )

    expect(unpriced.cost).toBeNull()
    expect(unpriced.cacheSaved).toBeNull()
    expect(unpriced.tokens.input).toBe(5)
  })

  it("measures the tool time a gateway's built-in tools spent inside the request that called them", () => {
    // The gateway's step runs one request, and its tools inside it.
    const step = (overrides: Parameters<typeof traceSpan>[0]) =>
      traceSpan({ parent_span_id: "step", ...overrides })
    const builtIn = sessionStats(
      traceSummary({ started_at: at(0), last_activity_at: at(45) }),
      traceDetail({
        spans: [
          traceSpan({
            span_id: "step",
            kind: "step",
            start_time: at(0),
            end_time: at(45),
          }),
          step({
            span_id: "r",
            kind: "llm",
            start_time: at(0),
            end_time: at(40),
          }),
          step({
            span_id: "search",
            kind: "tool",
            tool_name: "web_search",
            start_time: at(5),
            end_time: at(20),
          }),
          // Run past the request's end, so it was not run inside it.
          step({
            span_id: "run",
            kind: "tool",
            tool_name: "code_execution",
            start_time: at(35),
            end_time: at(45),
          }),
        ],
      }),
    )

    expect(builtIn.time.toolsMs).toBe(25_000)
    expect(builtIn.toolsInModelMs).toBe(15_000)
    expect(builtIn.activeMs).toBe(45_000)
  })

  it("does not count a tool as inside a request that ran beside it in another agent", () => {
    // Two subagents at once: B's Read falls inside A's open request.
    const parallel = sessionStats(
      traceSummary({ started_at: at(0), last_activity_at: at(60) }),
      traceDetail({
        spans: [
          traceSpan({
            span_id: "a-r",
            parent_span_id: "agent-a",
            kind: "llm",
            start_time: at(0),
            end_time: at(60),
          }),
          traceSpan({
            span_id: "b-read",
            parent_span_id: "agent-b",
            kind: "tool",
            tool_name: "Read",
            start_time: at(10),
            end_time: at(12),
          }),
        ],
      }),
    )

    expect(parallel.toolsInModelMs).toBe(0)
  })

  it("does not count a subagent's requests as tools inside a request", () => {
    // The Agent call's run spans its subagent's request, the reverse of a built-in tool.
    const delegated = sessionStats(
      traceSummary({ started_at: at(0), last_activity_at: at(60) }),
      traceDetail({
        spans: [
          traceSpan({
            span_id: "agent",
            kind: "tool",
            tool_name: "Agent",
            start_time: at(0),
            end_time: at(60),
          }),
          traceSpan({
            span_id: "sub",
            parent_span_id: "agent",
            kind: "llm",
            start_time: at(5),
            end_time: at(50),
          }),
        ],
      }),
    )

    expect(delegated.toolsInModelMs).toBe(0)
  })

  it("tallies each model's requests and what the gateway recorded for them, the costliest first", () => {
    const routed = sessionStats(
      traceSummary(),
      traceDetail({
        spans: [
          traceSpan({ span_id: "a", kind: "llm", model: "haiku", cost: 0.01 }),
          traceSpan({ span_id: "b", kind: "llm", model: "sonnet", cost: 0.2 }),
          traceSpan({ span_id: "c", kind: "llm", model: "haiku", cost: 0.02 }),
          traceSpan({ span_id: "d", kind: "llm", model: "free" }),
        ],
      }),
    )

    expect(routed.models).toEqual([
      { model: "sonnet", requests: 1, cost: 0.2 },
      { model: "haiku", requests: 2, cost: expect.closeTo(0.03) },
      { model: "free", requests: 1, cost: null },
    ])
  })

  it("reads a session whose spans name each other as parents", () => {
    const looped = [
      traceSpan({
        span_id: "a",
        parent_span_id: "b",
        kind: "tool",
        tool_name: "Bash",
      }),
      traceSpan({
        span_id: "b",
        parent_span_id: "a",
        kind: "other",
        operation: "tool.execution",
      }),
    ]

    const stats = sessionStats(traceSummary(), traceDetail({ spans: looped }))

    expect(stats.toolCalls).toBe(1)
  })
})
