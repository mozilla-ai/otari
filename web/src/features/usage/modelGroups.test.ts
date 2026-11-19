import { describe, expect, it } from "vitest"
import type { UsageGroupRow } from "@/client"
import { groupModels } from "@/features/usage/modelGroups"

const row = (overrides: Partial<UsageGroupRow>): UsageGroupRow => ({
  key: null,
  label: null,
  model_group: null,
  cost: 1,
  tokens: 10,
  requests: 1,
  is_other: false,
  ...overrides,
})

const FIREWORKS = row({
  key: "accounts/fireworks/models/deepseek-v4p1-flash-1017",
  label: "DeepSeek V4.1 Flash 1017",
  model_group: "deepseekv41flash1017",
  cost: 149.33,
  tokens: 8_590_300_000,
  requests: 900,
})
const NEBIUS = row({
  key: "deepseek-ai/DeepSeek-V4.1-Flash-1017",
  label: "DeepSeek V4.1 Flash 1017",
  model_group: "deepseekv41flash1017",
  cost: 50.47,
  tokens: 2_973_500_000,
  requests: 300,
})
const PRO = row({
  key: "deepseek-ai/DeepSeek-V4.1-Pro-1017",
  label: "DeepSeek V4.1 Pro 1017",
  model_group: "deepseekv41pro1017",
  cost: 63.57,
})

describe("groupModels", () => {
  it("sums one model's ids on two providers into one row, under its name", () => {
    const [flash, ...rest] = groupModels([FIREWORKS, PRO, NEBIUS])
    expect(flash).toEqual({
      id: "model:group:deepseekv41flash1017",
      name: "DeepSeek V4.1 Flash 1017",
      spellings: [FIREWORKS.key, NEBIUS.key],
      cost: FIREWORKS.cost + NEBIUS.cost,
      tokens: FIREWORKS.tokens + NEBIUS.tokens,
      requests: 1200,
      isOther: false,
    })
    // Another build of the family is another model.
    expect(rest.map((model) => model.name)).toEqual(["DeepSeek V4.1 Pro 1017"])
  })

  it("keeps the fold and usage that names no model apart from every model", () => {
    const models = groupModels([
      FIREWORKS,
      row({ is_other: true, cost: 3 }),
      row({ cost: 2 }),
    ])
    expect(
      models.map((model) => [model.id, model.name, model.spellings]),
    ).toEqual([
      [
        "model:group:deepseekv41flash1017",
        "DeepSeek V4.1 Flash 1017",
        [FIREWORKS.key],
      ],
      ["model::other", "Other", []],
      ["model::none", "Unknown model", []],
    ])
  })

  it("never folds the server's fold or unnamed usage into a model named like them", () => {
    const models = groupModels([
      row({ key: "other", label: "other", model_group: "other", cost: 4 }),
      row({ is_other: true, cost: 3 }),
      row({ key: "none", label: "none", model_group: "none", cost: 2 }),
      row({ cost: 1 }),
    ])
    expect(
      models.map((model) => [model.id, model.cost, model.isOther]),
    ).toEqual([
      ["model:group:other", 4, false],
      ["model::other", 3, true],
      ["model:group:none", 2, false],
      ["model::none", 1, false],
    ])
  })

  it("names usage recorded under an empty id as an unknown model", () => {
    expect(groupModels([row({ key: "", label: "" })])[0]?.name).toBe(
      "Unknown model",
    )
  })

  it("lists a row the gateway did not group under its own id", () => {
    const models = groupModels([
      row({ key: "gpt-5" }),
      row({ key: "gpt-5-mini" }),
    ])
    expect(models.map((model) => [model.id, model.name])).toEqual([
      ["model:id:gpt-5", "gpt-5"],
      ["model:id:gpt-5-mini", "gpt-5-mini"],
    ])
  })
})
