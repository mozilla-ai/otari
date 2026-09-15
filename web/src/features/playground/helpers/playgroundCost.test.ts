import { describe, expect, it } from "vitest"

import {
  computeTokensPerSecond,
  formatTurnCost,
  formatTurnDuration,
  formatTurnStats,
} from "./playgroundCost"
import type { TurnUsage } from "./playgroundTypes"

const usage: TurnUsage = {
  promptTokens: 1200,
  completionTokens: 34,
  cachedTokens: 0,
  costUsd: 0.0032,
  totalMs: 2300,
  ttftMs: 400,
  tokensPerSecond: 17.9,
}

describe("formatTurnCost", () => {
  it("says $0 for a free model rather than a rounded figure", () => {
    expect(formatTurnCost(0)).toBe("$0")
  })

  it("reports a bound below the smallest figure it can show", () => {
    // The failure this exists to avoid: "$0.0000" on a request that cost
    // money reads as free.
    expect(formatTurnCost(0.00002)).toBe("<$0.0001")
  })

  it("scales its precision to the magnitude", () => {
    expect(formatTurnCost(0.0032)).toBe("$0.0032")
    expect(formatTurnCost(0.42)).toBe("$0.420")
    expect(formatTurnCost(12.5)).toBe("$12.50")
  })
})

describe("formatTurnDuration", () => {
  it("uses seconds below a minute", () => {
    expect(formatTurnDuration(2300)).toBe("2.3s")
  })

  it("uses minutes and seconds above one", () => {
    expect(formatTurnDuration(65_000)).toBe("1m 5s")
  })

  it("carries a rounded-up remainder into the minute", () => {
    // Splitting first and rounding the remainder renders this as "1m 60s",
    // which is not a time.
    expect(formatTurnDuration(119_600)).toBe("2m 0s")
    expect(formatTurnDuration(59_500)).toBe("59.5s")
    expect(formatTurnDuration(60_500)).toBe("1m 1s")
  })

  it("reports a bound rather than no time at all", () => {
    expect(formatTurnDuration(4)).toBe("<0.1s")
  })
})

describe("computeTokensPerSecond", () => {
  it("divides output by the decode window", () => {
    expect(computeTokensPerSecond(40, 2000)).toBe(20)
  })

  it("is undefined with no output to measure", () => {
    expect(computeTokensPerSecond(0, 2000)).toBeUndefined()
  })

  it("is undefined with no window, rather than dividing by zero", () => {
    expect(computeTokensPerSecond(40, 0)).toBeUndefined()
  })
})

describe("formatTurnStats", () => {
  it("reads as one line of metadata", () => {
    expect(formatTurnStats(usage)).toBe(
      "1,200 in · 34 out · $0.0032 · 2.3s · 18 tok/s · 0.4s to first token",
    )
  })

  it("names cached tokens only when there were some", () => {
    expect(formatTurnStats({ ...usage, cachedTokens: 900 })).toContain(
      "900 cached",
    )
    expect(formatTurnStats(usage)).not.toContain("cached")
  })

  it("shortens rather than showing a placeholder cost", () => {
    // A deployment that prices nothing for a model still reports its tokens
    // and its timing; an empty "$" slot would read as free.
    const line = formatTurnStats({ ...usage, costUsd: undefined })
    expect(line).not.toContain("$")
    expect(line).toContain("34 out")
  })

  it("omits the first-token figure when nothing streamed", () => {
    expect(formatTurnStats({ ...usage, ttftMs: undefined })).not.toContain(
      "first token",
    )
  })
})
