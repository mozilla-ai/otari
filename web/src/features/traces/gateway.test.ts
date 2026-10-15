import { describe, expect, it } from "vitest"

import { traceDetail, traceSpan } from "@/tests/traceFixtures"

import { gatewayEvents, guardrailVerdict } from "./gateway"

const at = (seconds: number) =>
  new Date(Date.parse("2026-10-07T12:00:00Z") + seconds * 1000).toISOString()

function step(id: string, seconds: number, outcome: "ok" | "error" = "ok") {
  return traceSpan({ span_id: id, start_time: at(seconds), outcome })
}

function call(
  id: string,
  request: string,
  seconds: number,
  overrides: Parameters<typeof traceSpan>[0] = {},
) {
  return traceSpan({
    span_id: id,
    parent_span_id: request,
    request_id: request,
    kind: "llm",
    model: "gpt-5",
    provider: "openai",
    start_time: at(seconds),
    ...overrides,
  })
}

function guardrail(
  id: string,
  request: string,
  seconds: number,
  overrides: Parameters<typeof traceSpan>[0] = {},
) {
  return traceSpan({
    span_id: id,
    parent_span_id: request,
    request_id: request,
    kind: "guardrail",
    start_time: at(seconds),
    outcome: "error",
    error_class: "guardrail_flagged",
    attributes: {
      "otari.guardrail.profile": "pii",
      "otari.guardrail.mode": "block",
    },
    ...overrides,
  })
}

describe("guardrailVerdict", () => {
  it("reads a flag by its mode, and by the request when the mode is missing", () => {
    const monitor = guardrail("g", "r", 0, {
      attributes: { "otari.guardrail.mode": "monitor" },
    })
    const unnamed = guardrail("g", "r", 0, { attributes: {} })

    expect(guardrailVerdict(guardrail("g", "r", 0), step("r", 0))).toBe(
      "blocked",
    )
    expect(guardrailVerdict(monitor, step("r", 0))).toBe("flagged")
    expect(guardrailVerdict(unnamed, step("r", 0))).toBe("flagged")
    expect(guardrailVerdict(unnamed, step("r", 0, "error"))).toBe("blocked")
    expect(
      guardrailVerdict(
        guardrail("g", "r", 0, { outcome: "ok", error_class: null }),
        step("r", 0),
      ),
    ).toBe("passed")
  })

  it("says whether a guardrail that could not run stopped the request", () => {
    const unavailable = guardrail("g", "r", 0, {
      error_class: "guardrail_unavailable",
    })

    expect(guardrailVerdict(unavailable, step("r", 0, "error"))).toBe("refused")
    expect(guardrailVerdict(unavailable, step("r", 0))).toBe("unchecked")
  })
})

describe("gatewayEvents", () => {
  it("lists every place the gateway stepped in, in the order it happened", () => {
    const events = gatewayEvents(
      traceDetail({
        spans: [
          step("req-1", 0),
          guardrail("g-1", "req-1", 0, {
            attributes: {
              "otari.guardrail.profile": "pii",
              "otari.guardrail.mode": "monitor",
            },
          }),
          call("llm-1", "req-1", 1),
          step("req-2", 10),
          call("llm-2a", "req-2", 10, {
            outcome: "error",
            recovered: true,
            error_class: "status_429",
          }),
          call("llm-2b", "req-2", 11, {
            model: "claude-sonnet-4-5",
            provider: "anthropic",
          }),
          step("req-3", 20, "error"),
          guardrail("g-3", "req-3", 20),
          traceSpan({
            span_id: "mcp-3",
            parent_span_id: "req-3",
            kind: "mcp_connect",
            start_time: at(21),
            outcome: "error",
            error_class: "mcp_unreachable",
            attributes: { "otari.mcp.server": "github" },
          }),
          step("req-4", 30),
          call("llm-4", "req-4", 30, {
            attributes: {
              "otari.routing.policy": "cheapest",
              "otari.routing.selection_reason": "lowest_cost",
            },
          }),
        ],
      }),
    )

    expect(events.map((e) => [e.spanId, e.word])).toEqual([
      ["g-1", "Flagged"],
      ["llm-2a", "Fallback"],
      ["llm-2b", "Switched"],
      ["g-3", "Blocked"],
      ["mcp-3", "Unreachable"],
      ["llm-4", "Switched"],
    ])
    const [flagged, fallback, toClaude, blocked, mcp, back] = events
    expect(flagged.title).toBe("Flagged by the pii guardrail")
    expect(fallback.title).toBe("gpt-5 failed, claude-sonnet-4-5 answered")
    expect(fallback.detail).toBe(
      "Failed with HTTP 429. Routing tried the next candidate.",
    )
    expect(toClaude.title).toBe("Switched from gpt-5 to claude-sonnet-4-5")
    expect(toClaude.detail).toBe(
      "A fallback moved the session. The prompt cache built on gpt-5 does not carry over.",
    )
    expect(blocked).toMatchObject({
      tone: "danger",
      title: "Blocked by the pii guardrail",
    })
    expect(mcp.title).toBe("Could not reach the github MCP server")
    expect(back.title).toBe("Switched from claude-sonnet-4-5 to gpt-5")
    expect(back.detail).toBe(
      "Routing policy cheapest chose it (lowest cost). The prompt cache built on claude-sonnet-4-5 does not carry over.",
    )
  })

  it("reports nothing for a session the gateway only passed through", () => {
    expect(
      gatewayEvents(
        traceDetail({
          spans: [
            step("req-1", 0),
            guardrail("g-1", "req-1", 0, { outcome: "ok", error_class: null }),
            call("llm-1", "req-1", 1),
            call("llm-2", "req-1", 2),
          ],
        }),
      ),
    ).toEqual([])
  })

  it("ignores the model an instrumented agent reports for itself", () => {
    const events = gatewayEvents(
      traceDetail({
        spans: [
          call("llm-1", "req-1", 0),
          call("otlp-1", "req-1", 1, { origin: "otlp", model: "gpt-5-mini" }),
          call("llm-2", "req-2", 2),
        ],
      }),
    )

    expect(events).toEqual([])
  })
})
