import { describe, expect, it } from "vitest"

import {
  gateSummary,
  traceDetail,
  traceSpan,
  traceSummary,
} from "@/tests/traceFixtures"

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

  it("carries a call's cache and reasoning counts, and a turn's cached reads", () => {
    const detail = traceDetail()
    const spans = detail.spans.map((span) =>
      span.span_id === "llm-1"
        ? {
            ...span,
            cache_read_tokens: 6,
            cache_write_tokens: 2,
            reasoning_tokens: 1,
          }
        : span,
    )
    const turns = [
      { ...detail.turns[0], cache_read_tokens: 6 },
      { ...detail.turns[0], index: 1, step_ids: ["req-2"] },
    ]
    const trace = toTrace(traceSummary(), { ...detail, spans, turns })
    const byId = new Map(trace.observations.map((o) => [o.id, o]))

    expect(byId.get("llm-1")).toMatchObject({
      promptTokens: 10,
      cachedTokens: 6,
      cacheWriteTokens: 2,
      reasoningTokens: 1,
    })
    expect(byId.get("turn-0")?.cachedTokens).toBe(6)
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

  it("draws a tool a hook refused as blocked, with who refused it, not as an error", () => {
    const trace = toTrace(
      traceSummary(),
      traceDetail({
        spans: [
          traceSpan({ span_id: "req-1", origin: "otlp" }),
          traceSpan({
            span_id: "tool-1",
            parent_span_id: "req-1",
            kind: "tool",
            origin: "otlp",
            tool_name: "Edit",
            outcome: "blocked",
            attributes: {
              "otari.tool.decision": "reject",
              "otari.tool.decision_source": "hook",
            },
          }),
        ],
      }),
    )

    expect(trace.observations.find((o) => o.id === "tool-1")).toMatchObject({
      level: "WARNING",
      statusMessage: "Blocked by a hook before it ran.",
    })
  })

  it("marks a tool call that returned an error as a result the agent read, not an error", () => {
    const trace = toTrace(
      traceSummary(),
      traceDetail({
        spans: [
          traceSpan({ span_id: "req-1", origin: "otlp" }),
          traceSpan({
            span_id: "tool-1",
            parent_span_id: "req-1",
            kind: "tool",
            origin: "otlp",
            tool_name: "Bash",
            attributes: { "otari.tool.succeeded": false },
          }),
        ],
      }),
    )

    expect(trace.observations.find((o) => o.id === "tool-1")).toMatchObject({
      level: "DEFAULT",
      returnedError: true,
      statusMessage: expect.stringMatching(
        /^Returned an error, which the agent read/,
      ),
    })
  })

  it("draws a gate that blocked a tool call as a warning that explains it, not an error", () => {
    const trace = toTrace(
      traceSummary(),
      traceDetail({
        gates: [
          gateSummary({
            gate_id: "no-changelog",
            message: "CHANGELOG.md is generated.",
          }),
        ],
        spans: [
          traceSpan({ span_id: "req-1", origin: "otlp" }),
          traceSpan({
            span_id: "tool-1",
            parent_span_id: "req-1",
            kind: "tool",
            origin: "otlp",
            tool_name: "Edit",
            outcome: "blocked",
            attributes: {
              "otari.tool.decision": "reject",
              "otari.tool.decision_source": "hook",
            },
          }),
          traceSpan({
            span_id: "gate-1",
            parent_span_id: "tool-1",
            kind: "guardrail",
            origin: "otlp",
            outcome: "blocked",
            attributes: {
              "otari.guardrail.profile": "no-changelog",
              "otari.guardrail.mode": "block",
              "otari.gate.id": "no-changelog",
              "otari.gate.outcome": "fail",
              "otari.gate.blocked": true,
            },
          }),
          traceSpan({
            span_id: "gate-2",
            parent_span_id: "req-1",
            kind: "guardrail",
            origin: "otlp",
            attributes: {
              "otari.guardrail.profile": "docs-updated",
              "otari.guardrail.mode": "monitor",
              "otari.gate.id": "docs-updated",
              "otari.gate.outcome": "fail",
              "otari.gate.blocked": false,
            },
          }),
        ],
      }),
    )
    const byId = new Map(trace.observations.map((o) => [o.id, o]))

    expect(byId.get("tool-1")).toMatchObject({
      level: "WARNING",
      statusMessage:
        "Blocked by no-changelog before it ran. CHANGELOG.md is generated.",
    })
    expect(byId.get("tool-1:gates")).toMatchObject({
      name: "Gates · 1 blocked",
      parentObservationId: "tool-1",
      level: "WARNING",
    })
    expect(byId.get("req-1:gates")).toMatchObject({
      name: "Gates · 1 flagged",
    })
    expect(byId.get("gate-1")).toMatchObject({
      name: "no-changelog",
      parentObservationId: "tool-1:gates",
      level: "WARNING",
      statusMessage: "Blocked the agent's action. CHANGELOG.md is generated.",
    })
    expect(byId.get("gate-2")).toMatchObject({
      level: "WARNING",
      statusMessage: "Flagged. The gate is advisory, so the agent went ahead.",
    })
  })

  it("shows what a span was about as its input, where content was kept", () => {
    const trace = toTrace(
      traceSummary(),
      traceDetail({
        spans: [
          traceSpan({ span_id: "req-1", content: { prompt: "Fix the test" } }),
          traceSpan({
            span_id: "tool-1",
            parent_span_id: "req-1",
            kind: "tool",
            tool_name: "Bash",
            content: { command: "pytest -x" },
          }),
          traceSpan({
            span_id: "gate-1",
            parent_span_id: "tool-1",
            kind: "guardrail",
            content: { "gate.evidence": "pytest -x", "gate.detail": "ok" },
          }),
        ],
      }),
    )
    const byId = new Map(trace.observations.map((o) => [o.id, o]))

    expect(byId.get("req-1")?.input).toBe("Fix the test")
    expect(byId.get("tool-1")).toMatchObject({
      name: "Bash · pytest -x",
      input: "pytest -x",
    })
    expect(byId.get("gate-1")?.input).toEqual({
      "gate.evidence": "pytest -x",
      "gate.detail": "ok",
    })
  })

  it("gives every gate of a check what the check read, sent on one of them", () => {
    const gate = (span_id: string, gateId: string, otel_trace_id: string) =>
      traceSpan({
        span_id,
        parent_span_id: "tool-1",
        kind: "guardrail",
        origin: "otlp",
        otel_trace_id,
        attributes: {
          "otari.gate.id": gateId,
          "otari.gate.outcome": "pass",
          "otari.gate.blocked": false,
        },
      })
    const trace = toTrace(
      traceSummary(),
      traceDetail({
        spans: [
          traceSpan({ span_id: "req-1" }),
          traceSpan({
            span_id: "tool-1",
            parent_span_id: "req-1",
            kind: "tool",
          }),
          {
            ...gate("gate-1", "no-changelog", "check-a"),
            content: { "gate.evidence": "CHANGELOG.md" },
          },
          gate("gate-2", "no-lockfile", "check-a"),
          gate("gate-3", "no-changelog", "check-b"),
        ],
      }),
    )
    const byId = new Map(trace.observations.map((o) => [o.id, o]))

    expect(byId.get("gate-1")?.gate?.evidence).toBe("CHANGELOG.md")
    expect(byId.get("gate-2")?.gate?.evidence).toBe("CHANGELOG.md")
    expect(byId.get("gate-3")?.gate?.evidence).toBeNull()
  })

  it("says what a permission decision, a run and a model call were", () => {
    const trace = toTrace(
      traceSummary(),
      traceDetail({
        spans: [
          traceSpan({ span_id: "req-1" }),
          traceSpan({
            span_id: "tool-1",
            parent_span_id: "req-1",
            kind: "tool",
            tool_name: "Bash",
            content: { command: "git push" },
          }),
          traceSpan({
            span_id: "ask-1",
            parent_span_id: "tool-1",
            kind: "other",
            operation: "tool.blocked_on_user",
            duration_ms: 14_000,
            attributes: {
              "otari.tool.decision": "accept",
              "otari.tool.decision_source": "user_temporary",
            },
          }),
          traceSpan({
            span_id: "run-1",
            parent_span_id: "tool-1",
            kind: "other",
            operation: "tool.execution",
            duration_ms: 96,
          }),
          traceSpan({
            span_id: "llm-9",
            parent_span_id: "req-1",
            kind: "llm",
            model: "claude-haiku-4-5",
            attributes: { "gen_ai.response.finish_reasons": "tool_use" },
          }),
        ],
      }),
    )
    const byId = new Map(trace.observations.map((o) => [o.id, o]))

    expect(byId.get("ask-1")).toMatchObject({ name: "Permission · allowed" })
    expect(byId.get("ask-1")?.explain?.sentence).toBe(
      "Claude Code asked the user whether to run Bash on git push, and they allowed it after 14.00 s.",
    )
    expect(byId.get("run-1")).toMatchObject({ name: "Ran" })
    expect(byId.get("run-1")?.explain?.sentence).toMatch(
      /^Ran for .* and succeeded\.$/,
    )
    expect(byId.get("llm-9")?.explain?.facts).toContainEqual([
      "Stopped because",
      "It asked to use a tool",
    ])
  })

  it("says a tool call that returned an error ran and returned one, not that it succeeded", () => {
    const failed = { "otari.tool.succeeded": false }
    const trace = toTrace(
      traceSummary(),
      traceDetail({
        spans: [
          traceSpan({ span_id: "req-1" }),
          traceSpan({
            span_id: "tool-1",
            parent_span_id: "req-1",
            kind: "tool",
            tool_name: "Bash",
            content: { command: "grep -r nothing" },
            attributes: failed,
          }),
          traceSpan({
            span_id: "run-1",
            parent_span_id: "tool-1",
            kind: "other",
            operation: "tool.execution",
            duration_ms: 96,
            attributes: failed,
          }),
        ],
      }),
    )
    const byId = new Map(trace.observations.map((o) => [o.id, o]))

    expect(byId.get("run-1")?.explain?.sentence).toMatch(
      /^Ran for .* and returned an error\.$/,
    )
    expect(byId.get("tool-1")?.explain?.sentence).toMatch(
      /It ran and returned an error, which the agent read\.$/,
    )
  })

  it("says a model call that errored failed rather than answered", () => {
    const trace = toTrace(
      traceSummary(),
      traceDetail({
        spans: [
          traceSpan({ span_id: "req-1" }),
          traceSpan({
            span_id: "llm-1",
            parent_span_id: "req-1",
            kind: "llm",
            model: "claude-haiku-4-5",
            outcome: "error",
          }),
        ],
      }),
    )
    const llm = trace.observations.find((o) => o.id === "llm-1")

    expect(llm?.explain?.sentence).toMatch(/^claude-haiku-4-5 failed/)
  })

  it("names a gate that could not decide but held a tool call as what blocked it", () => {
    const trace = toTrace(
      traceSummary(),
      traceDetail({
        spans: [
          traceSpan({ span_id: "req-1" }),
          traceSpan({
            span_id: "tool-1",
            parent_span_id: "req-1",
            kind: "tool",
            tool_name: "Edit",
            outcome: "blocked",
          }),
          traceSpan({
            span_id: "gate-1",
            parent_span_id: "tool-1",
            kind: "guardrail",
            origin: "otlp",
            outcome: "error",
            attributes: {
              "otari.gate.id": "judge-review",
              "otari.gate.outcome": "error",
              "otari.gate.blocked": true,
            },
          }),
          traceSpan({
            span_id: "ask-1",
            parent_span_id: "tool-1",
            kind: "other",
            operation: "tool.blocked_on_user",
          }),
        ],
      }),
    )
    const byId = new Map(trace.observations.map((o) => [o.id, o]))

    expect(byId.get("tool-1")?.statusMessage).toBe(
      "Blocked by judge-review before it ran.",
    )
    expect(byId.get("tool-1:gates")?.name).toBe("Gates · 1 could not decide")
    expect(byId.get("ask-1")?.name).toBe("Permission")
    expect(byId.get("ask-1")?.explain?.sentence).toMatch(
      /did not say what was decided/,
    )
  })

  it("names the user for every user_* decision source Claude Code reports", () => {
    const trace = toTrace(
      traceSummary(),
      traceDetail({
        spans: [
          traceSpan({ span_id: "req-1", origin: "otlp" }),
          traceSpan({
            span_id: "tool-1",
            parent_span_id: "req-1",
            kind: "tool",
            origin: "otlp",
            tool_name: "Bash",
            outcome: "blocked",
            attributes: {
              "otari.tool.decision": "reject",
              "otari.tool.decision_source": "user_reject",
            },
          }),
        ],
      }),
    )

    expect(
      trace.observations.find((o) => o.id === "tool-1")?.statusMessage,
    ).toBe("Refused by the user.")
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
