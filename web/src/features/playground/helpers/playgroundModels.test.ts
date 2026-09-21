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

/** An indexed catalog of single-offering models: each id doubles as selector. */
function catalog(...ids: string[]): CatalogResponse {
  return catalogResponse(
    ids.map((id) => catalogModelSummary({ id, selectors: [`prov:${id}`] })),
  )
}

describe("isChatModel", () => {
  it("keeps a chat model", () => {
    expect(isChatModel("moonshotai/kimi-k3")).toBe(true)
  })

  it("drops the ids that are unambiguously not chat", () => {
    for (const id of [
      "text-embedding-3-large",
      "openai/whisper-1",
      "tts-1",
      "dall-e-3",
      "cohere/rerank-v3",
      "omni-moderation-latest",
    ]) {
      expect(isChatModel(id)).toBe(false)
    }
  })

  it("keeps an unfamiliar vendor's models", () => {
    // It errs toward keeping: a chat model wrongly hidden cannot be used and
    // says nothing about why, which is worse than one confusing refusal.
    expect(isChatModel("acme/model-7")).toBe(true)
  })

  it("ignores the vendor prefix, which says nothing about the model", () => {
    // A vendor named after a non-chat keyword must not hide its chat models.
    expect(isChatModel("guard-ai/chat-1")).toBe(true)
    expect(isChatModel("meta/llama-guard-3")).toBe(false)
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
    // The catalog arrives sorted by model name; re-sorting by key would break
    // that presentation.
    const models = buildPlaygroundModels(catalog("zeta", "alpha", "middle"))
    expect(models.map((model) => model.key)).toEqual([
      "zeta",
      "alpha",
      "middle",
    ])
  })

  it("offers a folded model once, keyed by its model-level selector", () => {
    // The Models page shows one card for GLM with two providers; the picker
    // shows one row too, and the gateway picks the offering.
    const models = buildPlaygroundModels(
      catalogResponse([
        catalogModelSummary({
          id: "z-ai/glm",
          selectors: ["fireworks:glm", "nebius:glm"],
        }),
      ]),
    )
    expect(models).toEqual([{ key: "z-ai/glm", vendor: "", label: "z-ai/glm" }])
  })

  it("falls back to the first offering while the catalog is unindexed", () => {
    // `selector` is null until the gateway has indexed the catalog; the model
    // must stay usable in the meantime.
    const models = buildPlaygroundModels(
      catalogResponse([
        catalogModelSummary({
          id: "z-ai/glm",
          selector: null,
          selectors: ["fireworks:glm", "nebius:glm"],
        }),
      ]),
    )
    expect(models.map((model) => model.key)).toEqual(["fireworks:glm"])
  })

  it("collapses a key the catalog lists twice", () => {
    expect(buildPlaygroundModels(catalog("gpt-4o", "gpt-4o"))).toHaveLength(1)
  })

  it("drops the non-chat entries", () => {
    expect(
      buildPlaygroundModels(catalog("gpt-4o", "text-embedding-3-small")).map(
        (model) => model.key,
      ),
    ).toEqual(["gpt-4o"])
  })

  it("is empty before the catalog answers", () => {
    expect(buildPlaygroundModels(undefined)).toEqual([])
  })
})

describe("groupPlaygroundModels", () => {
  const models = buildPlaygroundModels(
    catalogResponse([
      catalogModelSummary({
        id: "openai/gpt-4o",
        vendor: "OpenAI",
        selector: "openai/gpt-4o",
        selectors: ["prov:gpt-4o"],
      }),
      catalogModelSummary({
        id: "openai/gpt-4o-mini",
        vendor: "OpenAI",
        selector: "openai/gpt-4o-mini",
        selectors: ["prov:gpt-4o-mini"],
      }),
      catalogModelSummary({
        id: "anthropic/claude",
        vendor: "Anthropic",
        selector: "anthropic/claude",
        selectors: ["prov:claude"],
      }),
    ]),
  )

  it("groups by vendor, in first-seen order", () => {
    const groups = groupPlaygroundModels({
      models,
      pinnedKeys: [],
      search: "",
    })
    expect(groups.map((group) => group.label)).toEqual(["OpenAI", "Anthropic"])
  })

  it("leads with the pinned group", () => {
    const groups = groupPlaygroundModels({
      models,
      pinnedKeys: ["anthropic/claude"],
      search: "",
    })
    expect(groups[0]?.id).toBe(PINNED_GROUP_ID)
    expect(groups[0]?.models.map((model) => model.key)).toEqual([
      "anthropic/claude",
    ])
  })

  it("lists a pinned model once", () => {
    // A picker that shows the same row twice makes the second copy look like a
    // different model.
    const groups = groupPlaygroundModels({
      models,
      pinnedKeys: ["anthropic/claude"],
      search: "",
    })
    const everyKey = groups.flatMap((group) =>
      group.models.map((model) => model.key),
    )
    expect(everyKey.filter((key) => key === "anthropic/claude")).toHaveLength(1)
  })

  it("matches the whole key, so a vendor slug narrows to its models", () => {
    const groups = groupPlaygroundModels({
      models,
      pinnedKeys: [],
      search: "anthropic",
    })
    expect(groups).toHaveLength(1)
    expect(groups[0]?.label).toBe("Anthropic")
  })

  it("matches the displayed vendor, which the key may not contain", () => {
    // A bare-slug id like "gpt-4o" carries no vendor, but "OpenAI" is what the
    // group heading shows, so searching it must keep the group.
    const groups = groupPlaygroundModels({
      models: buildPlaygroundModels(
        catalogResponse([
          catalogModelSummary({
            id: "gpt-4o",
            vendor: "OpenAI",
            selector: "gpt-4o",
            selectors: ["prov:gpt-4o"],
          }),
        ]),
      ),
      pinnedKeys: [],
      search: "openai",
    })
    expect(groups).toHaveLength(1)
    expect(groups[0]?.models.map((model) => model.key)).toEqual(["gpt-4o"])
  })

  it("drops a group with no match rather than rendering it empty", () => {
    expect(
      groupPlaygroundModels({ models, pinnedKeys: [], search: "nothing" }),
    ).toEqual([])
  })

  it("labels an unknown vendor's group rather than leaving it blank", () => {
    const groups = groupPlaygroundModels({
      models: buildPlaygroundModels(catalog("gpt-4o")),
      pinnedKeys: [],
      search: "",
    })
    expect(groups[0]?.label).toBe("Other")
  })
})

describe("pickInitialModel", () => {
  const models = buildPlaygroundModels(catalog("gpt-4o", "anthropic/claude"))

  it("keeps the remembered model when it is still offered", () => {
    expect(pickInitialModel("anthropic/claude", models)).toBe(
      "anthropic/claude",
    )
  })

  it("falls back when the remembered model has left the catalog", () => {
    // A revoked key or a new restriction, and a picker showing a model the
    // gateway would refuse is worse than one that moved on.
    expect(pickInitialModel("gone/model", models)).toBe("gpt-4o")
  })

  it("is empty when there is nothing to pick", () => {
    expect(pickInitialModel(undefined, [])).toBe("")
  })
})
