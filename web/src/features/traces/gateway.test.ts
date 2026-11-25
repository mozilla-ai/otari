import { describe, expect, it } from "vitest"

import { traceDetail, traceSpan } from "@/tests/traceFixtures"

import { gateVerdict, gatewayEvents, guardrailVerdict } from "./gateway"

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

describe("gateVerdict", () => {
  const gate = (outcome: string, blocked: boolean) =>
    traceSpan({
      kind: "guardrail",
      origin: "otlp",
      attributes: {
        "otari.gate.outcome": outcome,
        "otari.gate.blocked": blocked,
      },
    })

  it("reads a reported gate's own verdict", () => {
    expect(gateVerdict(gate("pass", false))).toBe("passed")
    expect(gateVerdict(gate("fail", true))).toBe("blocked")
    expect(gateVerdict(gate("fail", false))).toBe("flagged")
    expect(gateVerdict(gate("unknown", true))).toBe("refused")
    expect(gateVerdict(gate("error", false))).toBe("unchecked")
  })

  it("is null for the gateway's own guardrails", () => {
    expect(gateVerdict(traceSpan({ kind: "guardrail" }))).toBeNull()
  })
})

describe("gatewayEvents, a fallback read from what followed it", () => {
  const failed = (
    id: string,
    seconds: number,
    errorClass: string,
    model: string,
    recovered = true,
  ) =>
    call(id, "req-1", seconds, {
      model,
      outcome: "error",
      recovered,
      error_class: errorClass,
    })
  const fallbacks = (spans: ReturnType<typeof call>[]) =>
    gatewayEvents(traceDetail({ spans: [step("req-1", 0), ...spans] })).filter(
      (e) => e.word === "Fallback",
    )

  it("says the next candidate answered when one did", () => {
    const [event] = fallbacks([
      failed("a", 1, "status_400", "gpt-5"),
      call("b", "req-1", 2, { model: "gpt-oss-120b" }),
    ])

    expect(event).toMatchObject({
      tone: "warning",
      title: "gpt-5 failed, gpt-oss-120b answered",
      detail: "Failed with HTTP 400. Routing tried the next candidate.",
    })
  })

  it("says the fallback failed too when two candidates all failed", () => {
    const events = fallbacks([
      failed("a", 1, "status_400", "gpt-5"),
      failed("b", 2, "status_504", "gpt-oss-120b", false),
    ])

    expect(events).toHaveLength(1)
    expect(events[0]).toMatchObject({
      tone: "danger",
      title: "gpt-5 failed, gpt-oss-120b failed too",
      detail:
        "Failed with HTTP 400. Routing tried gpt-oss-120b, which failed with HTTP 504. No candidate answered, so the request failed.",
    })
  })

  it("keeps each row true when three candidates all failed", () => {
    const events = fallbacks([
      failed("a", 1, "status_400", "gpt-5"),
      failed("b", 2, "status_429", "gpt-oss-120b"),
      failed("c", 3, "status_504", "claude-haiku", false),
    ])

    expect(events.map((e) => [e.tone, e.title, e.detail])).toEqual([
      [
        "danger",
        "gpt-5 failed, gpt-oss-120b failed too",
        "Failed with HTTP 400. Routing tried gpt-oss-120b, which failed with HTTP 429.",
      ],
      [
        "danger",
        "gpt-oss-120b failed, claude-haiku failed too",
        "Failed with HTTP 429. Routing tried claude-haiku, which failed with HTTP 504. No candidate answered, so the request failed.",
      ],
    ])
  })

  it("claims neither an answer nor a failure when no later call is visible", () => {
    const [event] = fallbacks([failed("a", 1, "status_400", "gpt-5")])

    expect(event).toMatchObject({
      tone: "warning",
      title: "gpt-5 failed",
      detail: "Failed with HTTP 400. Routing tried the next candidate.",
    })
  })

  it("makes no claim about a next call that was blocked, not failed", () => {
    const [event] = fallbacks([
      failed("a", 1, "status_400", "gpt-5"),
      call("b", "req-1", 2, {
        model: "gpt-oss-120b",
        outcome: "blocked",
        recovered: false,
      }),
    ])

    expect(event).toMatchObject({
      tone: "warning",
      title: "gpt-5 failed",
      detail: "Failed with HTTP 400. Routing tried the next candidate.",
    })
  })

  it("says no candidate answered only when every later call failed", () => {
    const events = fallbacks([
      failed("a", 1, "status_400", "gpt-5"),
      failed("b", 2, "status_429", "gpt-oss-120b"),
      call("c", "req-1", 3, { model: "claude-haiku", outcome: "blocked" }),
    ])

    expect(events[0].detail).toBe(
      "Failed with HTTP 400. Routing tried gpt-oss-120b, which failed with HTTP 429.",
    )
  })

  it("says no candidate answered only when the next call ended the request", () => {
    const [first] = fallbacks([
      failed("a", 1, "status_400", "gpt-5"),
      failed("b", 2, "status_429", "gpt-oss-120b"),
    ])

    expect(first.detail).toBe(
      "Failed with HTTP 400. Routing tried gpt-oss-120b, which failed with HTTP 429.",
    )
  })

  it("does not join calls that carry no request id", () => {
    const spans = [
      failed("a", 1, "status_400", "gpt-5"),
      call("b", "req-2", 2, { model: "gpt-oss-120b" }),
    ].map((span) => ({ ...span, request_id: null }))
    const [event] = fallbacks(spans)

    expect(event.title).toBe("gpt-5 failed")
  })

  it("reads a fallback that starts at the same instant as later", () => {
    const [event] = fallbacks([
      { ...failed("a", 1, "status_400", "gpt-5"), end_time: at(2) },
      call("b", "req-1", 1, {
        model: "gpt-oss-120b",
        end_time: at(3),
      }),
    ])

    expect(event.title).toBe("gpt-5 failed, gpt-oss-120b answered")
  })

  it("orders same-instant calls by their end, then by their position", () => {
    const events = fallbacks([
      failed("a", 1, "status_400", "gpt-5"),
      failed("b", 1, "status_429", "gpt-oss-120b"),
    ])

    expect(events.map((e) => e.title)).toEqual([
      "gpt-5 failed, gpt-oss-120b failed too",
      "gpt-oss-120b failed",
    ])
  })

  it("ignores later calls of another request", () => {
    const [event] = fallbacks([
      failed("a", 1, "status_400", "gpt-5"),
      call("b", "req-2", 2, { model: "gpt-oss-120b" }),
    ])

    expect(event.title).toBe("gpt-5 failed")
  })
})
