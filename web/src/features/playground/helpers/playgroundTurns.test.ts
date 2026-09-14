import { describe, expect, it } from "vitest"
import {
  appendErrorTurn,
  appendStreamDelta,
  patchLastAssistantTurn,
  shouldFollowToBottom,
  turnsForRegenerate,
  wireMessages,
} from "./playgroundTurns"
import type { ChatTurn } from "./playgroundTypes"

const question: ChatTurn = { role: "user", content: "hi" }
const answer: ChatTurn = { role: "assistant", content: "hello" }

describe("appendStreamDelta", () => {
  it("starts an assistant turn on the first fragment", () => {
    expect(appendStreamDelta([question], { content: "He" })).toEqual([
      question,
      { role: "assistant", content: "He", reasoning: undefined },
    ])
  })

  it("appends to the turn already in flight", () => {
    const started = appendStreamDelta([question], { content: "He" })
    expect(appendStreamDelta(started, { content: "llo" })[1]?.content).toBe(
      "Hello",
    )
  })

  it("accumulates content and reasoning independently", () => {
    let turns = appendStreamDelta([question], { reasoning: "think" })
    turns = appendStreamDelta(turns, { content: "answer" })
    turns = appendStreamDelta(turns, { reasoning: "ing" })
    expect(turns[1]).toMatchObject({
      content: "answer",
      reasoning: "thinking",
    })
  })

  it("never mutates the list it was given", () => {
    // The reason this is pure: two split-view panels fold their own streams at
    // once, and a mutation would make one panel's reducer visible in the other.
    const input = [question]
    appendStreamDelta(input, { content: "x" })
    expect(input).toEqual([question])
  })
})

describe("patchLastAssistantTurn", () => {
  it("replaces the trailing assistant turn", () => {
    const patched = patchLastAssistantTurn([question, answer], (last) => ({
      ...last,
      content: "patched",
    }))
    expect(patched[1]?.content).toBe("patched")
  })

  it("leaves a list ending in a question alone", () => {
    // A patch that arrives after a reset must not be written onto somebody's
    // own message.
    expect(patchLastAssistantTurn([question], () => answer)).toEqual([question])
  })
})

describe("appendErrorTurn", () => {
  it("attaches the failure to a partial reply, keeping what arrived", () => {
    const turns = appendErrorTurn(
      [question, { role: "assistant", content: "partial" }],
      "Upstream refused",
    )
    expect(turns[1]).toEqual({
      role: "assistant",
      content: "partial",
      errorMessage: "Upstream refused",
    })
  })

  it("adds a turn when nothing had streamed", () => {
    const turns = appendErrorTurn([question], "Upstream refused")
    expect(turns[1]).toEqual({
      role: "assistant",
      content: "",
      errorMessage: "Upstream refused",
    })
  })

  it("keeps the message out of the content", () => {
    // So a saved transcript does not store a gateway refusal as if the model
    // had said it.
    const [, errored] = appendErrorTurn([question], "Upstream refused")
    expect(errored?.content).not.toContain("Upstream refused")
  })
})

describe("turnsForRegenerate", () => {
  it("truncates back to the last question", () => {
    expect(turnsForRegenerate([question, answer])).toEqual([question])
  })

  it("drops every turn after that question", () => {
    const turns = [question, answer, { role: "user", content: "again" }, answer]
    expect(turnsForRegenerate(turns as ChatTurn[])).toEqual([
      question,
      answer,
      { role: "user", content: "again" },
    ])
  })

  it("returns an answer-only list unchanged, so the caller declines", () => {
    expect(turnsForRegenerate([answer])).toEqual([answer])
  })
})

describe("shouldFollowToBottom", () => {
  it("follows a question the reader just sent", () => {
    expect(
      shouldFollowToBottom({
        hasNewTurn: true,
        lastRole: "user",
        isPinnedToBottom: false,
      }),
    ).toBe(true)
  })

  it("does not follow a streamed token when they scrolled up to read", () => {
    expect(
      shouldFollowToBottom({
        hasNewTurn: false,
        lastRole: "assistant",
        isPinnedToBottom: false,
      }),
    ).toBe(false)
  })

  it("follows a streamed token when they are at the bottom", () => {
    expect(
      shouldFollowToBottom({
        hasNewTurn: false,
        lastRole: "assistant",
        isPinnedToBottom: true,
      }),
    ).toBe(true)
  })

  it("does not follow an answer that arrives while they are reading above", () => {
    // A new *assistant* turn is not a reason to jump: only their own question is.
    expect(
      shouldFollowToBottom({
        hasNewTurn: true,
        lastRole: "assistant",
        isPinnedToBottom: false,
      }),
    ).toBe(false)
  })
})

describe("wireMessages", () => {
  it("sends the role and the content and nothing else", () => {
    const turns: ChatTurn[] = [
      {
        role: "assistant",
        content: "hello",
        reasoning: "think",
        usage: {
          promptTokens: 1,
          completionTokens: 1,
          cachedTokens: 0,
          costUsd: 0,
          totalMs: 1,
          ttftMs: 1,
          tokensPerSecond: 1,
        },
      },
    ]
    expect(wireMessages(turns)).toEqual([
      { role: "assistant", content: "hello" },
    ])
  })
})
