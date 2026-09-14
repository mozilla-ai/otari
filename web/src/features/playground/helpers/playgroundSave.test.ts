import { describe, expect, it } from "vitest"

import {
  buildComparisonRequest,
  buildConversationRequest,
  buildConversationTitle,
  findRatedExchange,
  togglePinnedModel,
} from "./playgroundSave"
import type { ChatTurn } from "./playgroundTypes"

const exchange: ChatTurn[] = [
  { role: "user", content: "How does OAuth work?" },
  {
    role: "assistant",
    content: "It delegates authorization.",
    reasoning: "weighing it",
    usage: {
      promptTokens: 10,
      completionTokens: 5,
      cachedTokens: 0,
      costUsd: 0.001,
      totalMs: 100,
      ttftMs: 20,
      tokensPerSecond: 60,
    },
  },
]

describe("buildConversationTitle", () => {
  it("names the transcript after its first question", () => {
    expect(buildConversationTitle(exchange)).toBe("How does OAuth work?")
  })

  it("truncates a long question so the list stays scannable", () => {
    const title = buildConversationTitle([
      { role: "user", content: "x".repeat(200) },
    ])
    expect(title).toHaveLength(61)
    expect(title.endsWith("…")).toBe(true)
  })

  it("falls back rather than storing an unnamed row", () => {
    expect(
      buildConversationTitle([{ role: "assistant", content: "unprompted" }]),
    ).toBe("Untitled")
  })
})

describe("buildConversationRequest", () => {
  const request = buildConversationRequest({
    workspaceId: "ws-1",
    model: "openai:gpt-4o",
    turns: exchange,
  })

  it("keeps the reasoning, which a resumed transcript reads differently without", () => {
    expect(request.messages[1]?.reasoning).toBe("weighing it")
  })

  it("omits the field for a turn that had none, rather than sending null", () => {
    expect(request.messages[0]).not.toHaveProperty("reasoning")
  })

  it("stores no usage figures", () => {
    // They describe the request that ran, not the conversation: showing
    // yesterday's latency as this session's would be a lie.
    expect(request.messages[1]).not.toHaveProperty("usage")
  })

  it("carries the workspace and the model it ran on", () => {
    expect(request).toMatchObject({
      workspace_id: "ws-1",
      model: "openai:gpt-4o",
    })
  })
})

describe("findRatedExchange", () => {
  const answerA: ChatTurn = { role: "assistant", content: "A says" }
  const answerB: ChatTurn = { role: "assistant", content: "B says" }
  const question: ChatTurn = { role: "user", content: "which?" }

  it("finds the latest question and both answers to it", () => {
    expect(findRatedExchange([question, answerA], [question, answerB])).toEqual(
      { question: "which?", answerA: "A says", answerB: "B says" },
    )
  })

  it("anchors on the latest question, not the first", () => {
    const earlier: ChatTurn = { role: "user", content: "earlier" }
    expect(
      findRatedExchange(
        [earlier, answerA, question, answerA],
        [earlier, answerB, question, answerB],
      )?.question,
    ).toBe("which?")
  })

  it("is undefined while one panel has not answered", () => {
    // Which is what keeps the rating bar from recording half a comparison.
    expect(findRatedExchange([question, answerA], [question])).toBeUndefined()
  })

  it("is undefined when nothing was asked", () => {
    expect(findRatedExchange([], [])).toBeUndefined()
  })
})

describe("buildComparisonRequest", () => {
  it("stores both answers in full, which is what the consent covers", () => {
    expect(
      buildComparisonRequest({
        workspaceId: "ws-1",
        modelA: "openai:gpt-4o",
        modelB: "anthropic:claude",
        exchange: { question: "which?", answerA: "A", answerB: "B" },
        preference: "model_a",
      }),
    ).toEqual({
      workspace_id: "ws-1",
      user_question: "which?",
      model_a: "openai:gpt-4o",
      model_b: "anthropic:claude",
      model_a_answer: "A",
      model_b_answer: "B",
      preference: "model_a",
    })
  })
})

describe("togglePinnedModel", () => {
  it("puts a newly pinned model first", () => {
    // So the Pinned group is most-recent first: somebody pinning a model is
    // about to use it.
    expect(togglePinnedModel(["a"], "b")).toEqual(["b", "a"])
  })

  it("removes one that was already pinned", () => {
    expect(togglePinnedModel(["a", "b"], "a")).toEqual(["b"])
  })

  it("never mutates the list it was given", () => {
    const pinned = ["a"]
    togglePinnedModel(pinned, "b")
    expect(pinned).toEqual(["a"])
  })
})
