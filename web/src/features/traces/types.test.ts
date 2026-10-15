import { describe, expect, it } from "vitest"

import {
  type Observation,
  summarizeTrace,
  type Trace,
} from "@/features/traces/types"

function observation(overrides: Partial<Observation> = {}): Observation {
  return {
    id: "obs",
    parentObservationId: null,
    type: "SPAN",
    name: "span",
    startMs: 0,
    endMs: 10,
    level: "DEFAULT",
    ...overrides,
  }
}

const trace = (observations: Observation[]): Trace => ({
  id: "s-1",
  name: "claude-code",
  timestamp: "2026-10-07T12:00:00Z",
  observations,
})

describe("summarizeTrace", () => {
  it("measures the session to the latest end, not the last span's", () => {
    const summary = summarizeTrace(
      trace([
        observation({ id: "long", startMs: 0, endMs: 9_000 }),
        observation({ id: "late", startMs: 5_000, endMs: 6_000 }),
      ]),
    )

    expect(summary.latencyMs).toBe(9_000)
  })

  it("counts tokens from generations only, since a turn repeats them", () => {
    const summary = summarizeTrace(
      trace([
        observation({
          id: "turn",
          type: "SPAN",
          promptTokens: 15,
          completionTokens: 3,
        }),
        observation({
          id: "llm-1",
          parentObservationId: "turn",
          type: "GENERATION",
          promptTokens: 10,
          completionTokens: 2,
        }),
        observation({
          id: "llm-2",
          parentObservationId: "turn",
          type: "GENERATION",
          promptTokens: 5,
          completionTokens: 1,
        }),
      ]),
    )

    expect(summary).toMatchObject({
      promptTokens: 15,
      completionTokens: 3,
      totalTokens: 18,
    })
  })

  it("adds up every span's cost", () => {
    const summary = summarizeTrace(
      trace([
        observation({ id: "llm", type: "GENERATION", cost: 0.25 }),
        observation({ id: "guardrail", type: "GUARDRAIL", cost: 0.5 }),
        observation({ id: "tool", type: "TOOL" }),
      ]),
    )

    expect(summary.totalCost).toBe(0.75)
  })

  it("takes the worst level any span reached", () => {
    const levels = (...each: Observation["level"][]) =>
      summarizeTrace(
        trace(
          each.map((level, index) => observation({ id: `${index}`, level })),
        ),
      ).level

    expect(levels("DEFAULT", "DEFAULT")).toBe("DEFAULT")
    expect(levels("DEFAULT", "WARNING")).toBe("WARNING")
    expect(levels("WARNING", "ERROR", "DEFAULT")).toBe("ERROR")
    expect(levels("ERROR", "WARNING")).toBe("ERROR")
  })

  it("lists each model once", () => {
    const summary = summarizeTrace(
      trace([
        observation({ id: "1", type: "GENERATION", model: "gpt-5" }),
        observation({ id: "2", type: "GENERATION", model: "gpt-5" }),
        observation({ id: "3", type: "GENERATION", model: "claude-opus" }),
      ]),
    )

    expect(summary.models).toEqual(["gpt-5", "claude-opus"])
  })

  it("reads an empty session as nothing spent at the default level", () => {
    expect(summarizeTrace(trace([]))).toEqual({
      latencyMs: 0,
      promptTokens: 0,
      completionTokens: 0,
      totalTokens: 0,
      totalCost: 0,
      models: [],
      level: "DEFAULT",
    })
  })
})
