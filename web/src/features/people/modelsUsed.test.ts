import { describe, expect, it } from "vitest"
import { modelsUsed } from "@/features/people/modelsUsed"

describe("modelsUsed", () => {
  it("lists named models with tokens or spend, the costliest first", () => {
    const row = (
      key: string,
      cost: number,
      tokens: number,
      isOther = false,
    ) => ({
      key,
      label: key,
      cost,
      tokens,
      requests: 1,
      is_other: isOther,
    })
    expect(
      modelsUsed([
        row("cheap", 0.01, 900),
        row("idle", 0, 0),
        row("other", 5, 5, true),
        row("dear", 2, 10),
        row("free", 0, 50),
        row("tie", 0.01, 1000),
      ]).map((one) => one.name),
    ).toEqual(["dear", "tie", "cheap", "free"])
  })
})
