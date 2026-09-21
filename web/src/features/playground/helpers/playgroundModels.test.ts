import { describe, expect, it } from "vitest"

import type { CatalogResponse } from "@/client"
import { catalogModelSummary, catalogResponse } from "@/tests/fixtures"

import {
  buildPlaygroundModels,
  groupPlaygroundModels,
  isChatModel,
  PINNED_GROUP_ID,
  pickInitialModel,
  splitModelKey,
} from "./playgroundModels"

/** A grouped catalog of single-offering models, one per selector. */
function catalog(...selectors: string[]): CatalogResponse {
  return catalogResponse(
    selectors.map((selector) =>
      catalogModelSummary({ id: selector, selectors: [selector] }),
    ),
  )
}

describe("isChatModel", () => {
  it("keeps a chat model", () => {
    expect(isChatModel("openai:gpt-4o")).toBe(true)
  })

  it("drops the ids that are unambiguously not chat", () => {
    for (const id of [
      "openai:text-embedding-3-large",
      "openai:whisper-1",
      "openai:tts-1",
      "openai:dall-e-3",
      "cohere:rerank-v3",
      "openai:omni-moderation-latest",
    ]) {
      expect(isChatModel(id)).toBe(false)
    }
  })

  it("keeps an unfamiliar provider's models", () => {
    // It errs toward keeping: a chat model wrongly hidden cannot be used and
    // says nothing about why, which is worse than one confusing refusal.
    expect(isChatModel("acme:model-7")).toBe(true)
  })

  it("ignores the instance prefix, which says nothing about the model", () => {
    // An operator naming an instance "guard" or "embeddings" must not hide
    // that instance's whole chat catalog.
    expect(isChatModel("guard:gpt-4o")).toBe(true)
    expect(isChatModel("embeddings:claude")).toBe(true)
    expect(isChatModel("guard:llama-guard-3")).toBe(false)
  })
})

describe("splitModelKey", () => {
  it("splits an instance-addressed key", () => {
    expect(splitModelKey("openai:gpt-4o")).toEqual({
      instance: "openai",
      label: "gpt-4o",
    })
  })

  it("leaves a bare name without an instance", () => {
    expect(splitModelKey("gpt-4o")).toEqual({ instance: "", label: "gpt-4o" })
  })
})

describe("buildPlaygroundModels", () => {
  it("keeps the catalog's own order", () => {
    // The catalog arrives sorted by model name; re-sorting by selector would
    // split offerings of one model away from each other.
    const models = buildPlaygroundModels(
      catalog("zeta:a", "alpha:b", "middle:c"),
    )
    expect(models.map((model) => model.key)).toEqual([
      "zeta:a",
      "alpha:b",
      "middle:c",
    ])
  })

  it("offers every offering of a model the catalog folded", () => {
    // The Models page shows one card for GLM with two providers; the picker
    // shows both selectors, because that is what a request addresses.
    const models = buildPlaygroundModels(
      catalogResponse([
        catalogModelSummary({
          id: "z-ai/glm",
          selectors: ["fireworks:glm", "nebius:glm"],
        }),
      ]),
    )
    expect(models.map((entry) => entry.key)).toEqual([
      "fireworks:glm",
      "nebius:glm",
    ])
  })

  it("collapses a selector the catalog lists twice", () => {
    expect(
      buildPlaygroundModels(catalog("openai:gpt-4o", "openai:gpt-4o")),
    ).toHaveLength(1)
  })

  it("drops the non-chat entries", () => {
    expect(
      buildPlaygroundModels(
        catalog("openai:gpt-4o", "openai:text-embedding-3-small"),
      ).map((model) => model.key),
    ).toEqual(["openai:gpt-4o"])
  })

  it("is empty before the catalog answers", () => {
    expect(buildPlaygroundModels(undefined)).toEqual([])
  })
})

describe("groupPlaygroundModels", () => {
  const models = buildPlaygroundModels(
    catalog("openai:gpt-4o", "openai:gpt-4o-mini", "anthropic:claude"),
  )

  it("groups by provider instance, in first-seen order", () => {
    const groups = groupPlaygroundModels({
      models,
      pinnedKeys: [],
      search: "",
    })
    expect(groups.map((group) => group.label)).toEqual(["openai", "anthropic"])
  })

  it("leads with the pinned group", () => {
    const groups = groupPlaygroundModels({
      models,
      pinnedKeys: ["anthropic:claude"],
      search: "",
    })
    expect(groups[0]?.id).toBe(PINNED_GROUP_ID)
    expect(groups[0]?.models.map((model) => model.key)).toEqual([
      "anthropic:claude",
    ])
  })

  it("lists a pinned model once", () => {
    // A picker that shows the same row twice makes the second copy look like a
    // different model.
    const groups = groupPlaygroundModels({
      models,
      pinnedKeys: ["anthropic:claude"],
      search: "",
    })
    const everyKey = groups.flatMap((group) =>
      group.models.map((model) => model.key),
    )
    expect(everyKey.filter((key) => key === "anthropic:claude")).toHaveLength(1)
  })

  it("matches the whole key, so a provider name narrows to its instance", () => {
    const groups = groupPlaygroundModels({
      models,
      pinnedKeys: [],
      search: "anthropic",
    })
    expect(groups).toHaveLength(1)
    expect(groups[0]?.label).toBe("anthropic")
  })

  it("drops a group with no match rather than rendering it empty", () => {
    expect(
      groupPlaygroundModels({ models, pinnedKeys: [], search: "nothing" }),
    ).toEqual([])
  })

  it("labels a bare model name's group rather than leaving it blank", () => {
    const groups = groupPlaygroundModels({
      models: buildPlaygroundModels(catalog("gpt-4o")),
      pinnedKeys: [],
      search: "",
    })
    expect(groups[0]?.label).toBe("Other")
  })
})

describe("pickInitialModel", () => {
  const models = buildPlaygroundModels(
    catalog("openai:gpt-4o", "anthropic:claude"),
  )

  it("keeps the remembered model when it is still offered", () => {
    expect(pickInitialModel("anthropic:claude", models)).toBe(
      "anthropic:claude",
    )
  })

  it("falls back when the remembered model has left the catalog", () => {
    // A revoked key or a new restriction, and a picker showing a model the
    // gateway would refuse is worse than one that moved on.
    expect(pickInitialModel("gone:model", models)).toBe("openai:gpt-4o")
  })

  it("is empty when there is nothing to pick", () => {
    expect(pickInitialModel(undefined, [])).toBe("")
  })
})
