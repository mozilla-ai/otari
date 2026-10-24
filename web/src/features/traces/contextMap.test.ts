import { describe, expect, it } from "vitest"

import {
  buildContextMap,
  type ContextWeights,
  contextWeightsOf,
  formatBlockTokens,
  formatShare,
  formatUsedPercent,
  formatWindow,
  scaleToTotal,
} from "./contextMap"

const WEIGHTS: ContextWeights = {
  system: 4_000,
  tools: 12_000,
  user: 3_000,
  assistant: 5_000,
  toolResults: 30_000,
  files: 4_000,
  toolCount: 12,
  messageCount: 41,
}

function sum(values: number[]) {
  return values.reduce((a, b) => a + b, 0)
}

describe("scaleToTotal", () => {
  it("scales weights so they add up to the total exactly", () => {
    for (const total of [1, 7, 99, 1_000, 39_600, 123_457]) {
      for (const weights of [
        [1, 1, 1],
        [3, 5, 7, 11],
        [1, 0, 2],
        [4_000, 12_000, 8_000, 30_000, 4_000],
      ]) {
        expect(sum(scaleToTotal(weights, total))).toBe(total)
      }
    }
  })

  it("hands what rounding left to the largest remainders", () => {
    // 10 / 3 = 3.33 each: one extra token, to the first of the equal remainders.
    expect(scaleToTotal([1, 1, 1], 10)).toEqual([4, 3, 3])
    // 5 * 0.7 = 3.5 and 5 * 0.3 = 1.5: both remainders tie, so the larger weight wins.
    expect(scaleToTotal([7, 3], 5)).toEqual([4, 1])
    // 0.9 and 0.1 of 11 are 9.9 and 1.1: the larger remainder gets the token.
    expect(scaleToTotal([9, 1], 11)).toEqual([10, 1])
  })

  it("keeps a part with no weight at none", () => {
    expect(scaleToTotal([0, 3, 0, 1], 100)).toEqual([0, 75, 0, 25])
  })

  it("gives nothing when there is nothing to scale", () => {
    expect(scaleToTotal([0, 0], 10)).toEqual([0, 0])
    expect(scaleToTotal([1, 2], 0)).toEqual([0, 0])
    expect(scaleToTotal([], 5)).toEqual([])
  })
})

describe("formatting", () => {
  it("writes counts as plain integers, k with one decimal, and M", () => {
    expect(formatBlockTokens(0)).toBe("0")
    expect(formatBlockTokens(999)).toBe("999")
    expect(formatBlockTokens(1_000)).toBe("1.0k")
    expect(formatBlockTokens(14_800)).toBe("14.8k")
    expect(formatBlockTokens(39_600)).toBe("39.6k")
    expect(formatBlockTokens(1_250_000)).toBe("1.3M")
    expect(formatBlockTokens(2_000_000)).toBe("2.0M")
  })

  it("promotes a count that rounds to a thousand k to M", () => {
    expect(formatBlockTokens(999_960)).toBe("1.0M")
  })

  it("writes a window without a trailing .0", () => {
    expect(formatWindow(200_000)).toBe("200k")
    expect(formatWindow(1_000_000)).toBe("1M")
    expect(formatWindow(128_000)).toBe("128k")
    expect(formatWindow(8_192)).toBe("8.2k")
    expect(formatWindow(512)).toBe("512")
  })

  it("writes a tooltip share to one decimal and the header's to a whole number", () => {
    expect(formatShare(0.088)).toBe("8.8%")
    expect(formatShare(0.1)).toBe("10.0%")
    expect(formatUsedPercent(39_600, 200_000)).toBe("20%")
    expect(formatUsedPercent(100_000, 200_000)).toBe("50%")
    expect(formatUsedPercent(100, 200_000)).toBe("<1%")
    expect(formatUsedPercent(250_000, 200_000)).toBe(">100%")
    expect(formatUsedPercent(200_000, 200_000)).toBe("100%")
  })
})

describe("contextWeightsOf", () => {
  it("reads the numbers a gateway-served request kept", () => {
    expect(
      contextWeightsOf({
        "otari.context.system_chars": 10,
        "otari.context.tools_chars": 20,
        "otari.context.user_chars": 30,
        "otari.context.assistant_chars": 40,
        "otari.context.tool_result_chars": 50,
        "otari.context.file_chars": 60,
        "otari.context.tool_count": 3,
        "otari.context.message_count": 4,
        "gen_ai.request.max_tokens": 1024,
      }),
    ).toEqual({
      system: 10,
      tools: 20,
      user: 30,
      assistant: 40,
      toolResults: 50,
      files: 60,
      toolCount: 3,
      messageCount: 4,
    })
  })

  it("is undefined for a span that kept none, or kept something that is not a count", () => {
    expect(contextWeightsOf({})).toBeUndefined()
    expect(
      contextWeightsOf({ "otari.context.user_chars": "12" }),
    ).toBeUndefined()
    expect(contextWeightsOf({ "otari.context.user_chars": -4 })).toBeUndefined()
  })
})

describe("buildContextMap", () => {
  it("scales an estimated composition to the exact input tokens", () => {
    const map = buildContextMap({
      inputTokens: 39_600,
      windowTokens: 200_000,
      weights: WEIGHTS,
    })

    expect(map).toBeDefined()
    expect(map?.isEstimate).toBe(true)
    const used = map?.blocks.filter((block) => block.kind !== "free") ?? []
    expect(sum(used.map((block) => block.tokens))).toBe(39_600)
    expect(used.map((block) => block.kind)).toEqual([
      "system",
      "tools",
      "files",
      "messages",
      "toolResults",
    ])
    expect(used.find((block) => block.kind === "messages")?.tokens).toBe(
      Math.round((8_000 / 58_000) * 39_600),
    )
  })

  it("adds the rest of the window as Free", () => {
    const map = buildContextMap({
      inputTokens: 39_600,
      windowTokens: 200_000,
      weights: WEIGHTS,
    })
    const free = map?.blocks.find((block) => block.kind === "free")

    expect(free).toMatchObject({ tokens: 160_400, title: "Free" })
    expect(free?.share).toBeCloseTo(0.802)
    expect(sum(map?.blocks.map((block) => block.tokens) ?? [])).toBe(200_000)
    expect(
      map?.blocks.find((block) => block.kind === "tools")?.share,
    ).toBeCloseTo(((12_000 / 58_000) * 39_600) / 200_000, 3)
  })

  it("says how many tool definitions and messages the tooltip is about", () => {
    const map = buildContextMap({ inputTokens: 1_000, weights: WEIGHTS })

    expect(
      map?.blocks.find((block) => block.kind === "tools")?.notes,
    ).toContain("12 tool definitions")
    expect(
      map?.blocks.find((block) => block.kind === "messages")?.notes,
    ).toContain("41 messages")
    expect(
      buildContextMap({
        inputTokens: 1_000,
        weights: { ...WEIGHTS, toolCount: 1, messageCount: 1 },
      })?.blocks.flatMap((block) => block.notes),
    ).toEqual(expect.arrayContaining(["1 tool definition", "1 message"]))
  })

  it("leaves out a part that weighs nothing", () => {
    const map = buildContextMap({
      inputTokens: 100,
      weights: {
        ...WEIGHTS,
        system: 0,
        tools: 0,
        files: 0,
        toolResults: 0,
        toolCount: 0,
      },
    })

    expect(map?.blocks.map((block) => block.kind)).toEqual(["messages"])
    expect(map?.blocks[0].tokens).toBe(100)
  })

  it("falls back to the cache split when no weights were kept", () => {
    const map = buildContextMap({
      inputTokens: 48_210,
      cacheReadTokens: 41_900,
      cacheWriteTokens: 6_010,
      windowTokens: 200_000,
    })

    expect(map?.isEstimate).toBe(false)
    expect(map?.blocks.map((block) => [block.kind, block.tokens])).toEqual([
      ["cacheRead", 41_900],
      ["cacheWrite", 6_010],
      ["fresh", 300],
      ["free", 151_790],
    ])
  })

  it("falls back to the cache split when the weights are all zero", () => {
    const empty = {
      ...WEIGHTS,
      system: 0,
      tools: 0,
      user: 0,
      assistant: 0,
      toolResults: 0,
      files: 0,
    }

    expect(
      buildContextMap({
        inputTokens: 50,
        cacheReadTokens: 20,
        weights: empty,
      })?.blocks.map((block) => block.kind),
    ).toEqual(["cacheRead", "fresh"])
  })

  it("draws one block of input where the cache counts are unknown", () => {
    const map = buildContextMap({ inputTokens: 500 })

    expect(map?.blocks).toHaveLength(1)
    expect(map?.blocks[0]).toMatchObject({
      kind: "fresh",
      title: "Input",
      tokens: 500,
      share: 1,
    })
  })

  it("never lets the cache counts add up to more than the input", () => {
    const map = buildContextMap({
      inputTokens: 100,
      cacheReadTokens: 90,
      cacheWriteTokens: 90,
    })

    expect(sum(map?.blocks.map((block) => block.tokens) ?? [])).toBe(100)
  })

  it("shows no Free block and takes shares of the input where the window is unknown", () => {
    const map = buildContextMap({
      inputTokens: 1_000,
      cacheReadTokens: 250,
      cacheWriteTokens: 0,
    })

    expect(map?.windowTokens).toBeUndefined()
    expect(map?.blocks.some((block) => block.kind === "free")).toBe(false)
    expect(map?.blocks.map((block) => block.share)).toEqual([0.25, 0.75])
    expect(
      buildContextMap({ inputTokens: 10, windowTokens: 0 })?.windowTokens,
    ).toBeUndefined()
  })

  it("caps an input larger than the window: no Free, shares of the input", () => {
    const map = buildContextMap({
      inputTokens: 250_000,
      windowTokens: 200_000,
      cacheReadTokens: 100_000,
    })

    expect(map?.isOverWindow).toBe(true)
    expect(map?.blocks.some((block) => block.kind === "free")).toBe(false)
    expect(sum(map?.blocks.map((block) => block.share) ?? [])).toBeCloseTo(1)
  })

  it("has no Free block for an input that fills the window exactly", () => {
    const map = buildContextMap({ inputTokens: 200_000, windowTokens: 200_000 })

    expect(map?.isOverWindow).toBe(false)
    expect(map?.blocks.some((block) => block.kind === "free")).toBe(false)
  })

  it("has no map for a request with no input tokens", () => {
    expect(buildContextMap({})).toBeUndefined()
    expect(
      buildContextMap({ inputTokens: 0, windowTokens: 200_000 }),
    ).toBeUndefined()
  })
})
