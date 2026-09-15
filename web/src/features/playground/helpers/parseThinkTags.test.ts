import { describe, expect, it } from "vitest"

import { parseThinkTags } from "./parseThinkTags"

describe("parseThinkTags", () => {
  it("reports no reasoning for an ordinary reply", () => {
    expect(parseThinkTags("just the answer")).toEqual({
      thinking: undefined,
      response: "just the answer",
    })
  })

  it("splits a closed block out of the body", () => {
    expect(parseThinkTags("<think>weighing it</think>the answer")).toEqual({
      thinking: "weighing it",
      response: "the answer",
    })
  })

  it("treats an unclosed block as reasoning that is still arriving", () => {
    // The streaming case, and the whole reason this is not a regex replace: at
    // this instant the tag has opened and not closed, and without this the raw
    // `<think>` and the partial reasoning flash into the answer and then
    // vanish when a later chunk closes them.
    expect(parseThinkTags("<think>weighing i")).toEqual({
      thinking: "weighing i",
      response: "",
    })
  })

  it("joins several blocks and keeps none of them in the body", () => {
    // What a tool-using turn looks like: one completion whose content holds
    // reasoning before the tool call and more after the result.
    expect(
      parseThinkTags("<think>before</think>mid<think>after</think>end"),
    ).toEqual({ thinking: "before\n\nafter", response: "midend" })
  })

  it("keeps text that precedes the first block", () => {
    expect(parseThinkTags("lead <think>why</think> tail")).toEqual({
      thinking: "why",
      response: "lead  tail",
    })
  })
})
