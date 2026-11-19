import { describe, expect, it } from "vitest"
import type { UsageGroupRow } from "@/client"
import { pickTopModel, topModelShare } from "@/features/usage/topModel"

const row = (overrides: Partial<UsageGroupRow>): UsageGroupRow => ({
  key: "m",
  cost: 0,
  tokens: 0,
  requests: 0,
  is_other: false,
  ...overrides,
})

describe("pickTopModel", () => {
  it("takes the model with the most spend", () => {
    const top = pickTopModel([
      row({ key: "small", cost: 1 }),
      row({ key: "big", cost: 9 }),
    ])
    expect(top).toEqual({ name: "big", spellings: ["big"], cost: 9 })
  })

  it("breaks a tie on spend by tokens", () => {
    const top = pickTopModel([
      row({ key: "few", cost: 2, tokens: 10 }),
      row({ key: "many", cost: 2, tokens: 500 }),
    ])
    expect(top?.name).toBe("many")
  })

  it("counts a model's ids on every provider as one model, by its name", () => {
    const top = pickTopModel([
      row({ key: "solo", cost: 5 }),
      row({
        key: "accounts/fireworks/models/glm-5p3",
        label: "GLM 5.3",
        model_group: "z-ai/glm-5.3",
        cost: 3,
      }),
      row({
        key: "zai-org/GLM-5.3",
        label: "GLM 5.3",
        model_group: "z-ai/glm-5.3",
        cost: 3,
      }),
    ])
    expect(top).toEqual({
      name: "GLM 5.3",
      spellings: ["accounts/fireworks/models/glm-5p3", "zai-org/GLM-5.3"],
      cost: 6,
    })
  })

  it("never picks the fold of other models", () => {
    const top = pickTopModel([
      row({ key: null, cost: 50, is_other: true }),
      row({ key: "named", cost: 3 }),
    ])
    expect(top?.name).toBe("named")
  })

  it("names a model with no key", () => {
    expect(pickTopModel([row({ key: null, cost: 4 })])?.name).toBe(
      "Unknown model",
    )
  })

  it("is undefined with nothing to pick from", () => {
    expect(pickTopModel([])).toBeUndefined()
    expect(
      pickTopModel([row({ key: null, cost: 5, is_other: true })]),
    ).toBeUndefined()
  })

  it("does not reorder its input", () => {
    const rows = [row({ key: "a", cost: 1 }), row({ key: "b", cost: 2 })]
    pickTopModel(rows)
    expect(rows.map((one) => one.key)).toEqual(["a", "b"])
  })
})

describe("topModelShare", () => {
  it("reads the share of the row's spend", () => {
    expect(
      topModelShare({ name: "m", spellings: ["m"], cost: 32 }, 33.26),
    ).toBe("96% of spend")
  })

  it("says unpriced when the row has no spend", () => {
    expect(topModelShare({ name: "m", spellings: ["m"], cost: 0 }, 0)).toBe(
      "unpriced",
    )
  })
})
