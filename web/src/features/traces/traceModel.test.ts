import { describe, expect, it } from "vitest"
import {
  buildTraceTree,
  sessionName,
  spanLabel,
  spanTone,
  traceKind,
  turnStatus,
} from "@/features/traces/traceModel"
import { traceDetail, traceSpan, traceSummary } from "@/tests/traceFixtures"

describe("buildTraceTree", () => {
  it("puts each request under its turn and everything it ran under the request", () => {
    const tree = buildTraceTree(traceDetail())

    expect(tree.turns).toHaveLength(1)
    const [first, second] = tree.turns[0].steps
    expect(first.span.span_id).toBe("req-1")
    expect(first.children.map((node) => node.span.span_id)).toEqual([
      "llm-1",
      "call-toolu_1",
    ])
    expect(second.children.map((node) => node.span.span_id)).toEqual(["llm-2"])
    expect(tree.roots).toEqual([])
  })

  it("keeps an instrumented agent's own spans as roots with their tree", () => {
    const tree = buildTraceTree(
      traceDetail({
        turns: [],
        spans: [
          traceSpan({
            span_id: "otlp-a",
            kind: "agent",
            name: "invoke_agent",
            origin: "otlp",
          }),
          traceSpan({
            span_id: "otlp-t",
            parent_span_id: "otlp-a",
            kind: "tool",
            tool_name: "searchDocs",
          }),
        ],
      }),
    )

    expect(tree.roots.map((node) => node.span.span_id)).toEqual(["otlp-a"])
    expect(tree.roots[0].children.map((node) => node.span.span_id)).toEqual([
      "otlp-t",
    ])
  })
})

describe("labels", () => {
  it("names a span by what it acted on", () => {
    expect(spanLabel(traceSpan({ kind: "llm", model: "gpt-5" }))).toBe("gpt-5")
    expect(spanLabel(traceSpan({ kind: "tool", tool_name: "Bash" }))).toBe(
      "Bash",
    )
    expect(spanLabel(traceSpan({ kind: "step" }))).toBe("Request")
  })

  it("tells a recovered failure from one nothing made up for", () => {
    expect(spanTone(traceSpan({ outcome: "error", recovered: true }))).toBe(
      "warning",
    )
    expect(spanTone(traceSpan({ outcome: "error" }))).toBe("danger")
  })

  it("names a session by its agent, or as a single request", () => {
    expect(sessionName(traceSummary())).toBe("claude-code")
    expect(
      sessionName(traceSummary({ harness: null, session_source: "none" })),
    ).toBe("Request")
  })

  it("maps each turn state to a status", () => {
    expect(turnStatus(traceDetail().turns[0])).toEqual({
      tone: "success",
      label: "Completed",
    })
  })
})

describe("traceKind", () => {
  it("calls a trace that named a session a session, and one that named none a request", () => {
    expect(traceKind(traceSummary({ session_source: "harness" }))).toBe(
      "Session",
    )
    expect(traceKind(traceSummary({ session_source: "none" }))).toBe("Request")
  })
})
