import { describe, expect, it } from "vitest"

import { matchesSearch } from "@/features/traces/search"
import type { Observation } from "@/features/traces/types"

const observation: Observation = {
  id: "span-7f3a",
  parentObservationId: null,
  type: "GENERATION",
  name: "chat completion",
  startMs: 0,
  endMs: 10,
  level: "DEFAULT",
}

describe("matchesSearch", () => {
  it("matches the name, the type or the id, ignoring case", () => {
    expect(matchesSearch(observation, "Completion")).toBe(true)
    expect(matchesSearch(observation, "generation")).toBe(true)
    expect(matchesSearch(observation, "7F3A")).toBe(true)
  })

  it("trims the query, so a pasted id with whitespace still finds its span", () => {
    expect(matchesSearch(observation, "  span-7f3a\n")).toBe(true)
  })

  it("matches everything on a blank query", () => {
    expect(matchesSearch(observation, "   ")).toBe(true)
  })

  it("rejects a query none of the fields contain", () => {
    expect(matchesSearch(observation, "tool")).toBe(false)
  })
})
