import { afterEach, describe, expect, it } from "vitest"

import {
  publicCatalogHref,
  publicCatalogPath,
  rememberModel,
  takeRememberedModel,
} from "@/features/models/publicCatalog"

describe("publicCatalogPath", () => {
  it("names the list and a selected model", () => {
    expect(publicCatalogPath("#/models")).toEqual({ modelId: undefined })
    expect(publicCatalogPath("#/models/")).toEqual({ modelId: undefined })
    expect(publicCatalogPath("#/models/z-ai/glm-5.3")).toEqual({
      modelId: "z-ai/glm-5.3",
    })
  })

  it("ignores a query string and refuses everything else", () => {
    expect(publicCatalogPath("#/models?provider=nebius")).toEqual({
      modelId: undefined,
    })
    expect(publicCatalogPath("#/")).toBeNull()
    expect(publicCatalogPath("#/keys")).toBeNull()
    // A malformed percent escape is not a model id, and decoding one throws.
    expect(publicCatalogPath("#/models/%")).toBeNull()
  })

  it("round-trips through the href", () => {
    expect(
      publicCatalogPath(publicCatalogHref("moonshotai/kimi-k2.6")),
    ).toEqual({
      modelId: "moonshotai/kimi-k2.6",
    })
    expect(publicCatalogHref()).toBe("#/models")
  })
})

describe("the remembered model", () => {
  afterEach(() => window.localStorage.clear())

  it("is handed back once, then forgotten", () => {
    rememberModel("moonshotai/kimi-k3", 1_000)
    expect(takeRememberedModel(2_000)).toBe("moonshotai/kimi-k3")
    expect(takeRememberedModel(2_000)).toBeNull()
  })

  it("goes stale after a day, and is forgotten all the same", () => {
    rememberModel("moonshotai/kimi-k3", 0)
    expect(takeRememberedModel(24 * 60 * 60 * 1000 + 1)).toBeNull()
    expect(
      window.localStorage.getItem("otari.catalog.remembered-model"),
    ).toBeNull()
  })

  it("reads anything else in the slot as nothing", () => {
    window.localStorage.setItem("otari.catalog.remembered-model", "not json")
    expect(takeRememberedModel()).toBeNull()
    window.localStorage.setItem(
      "otari.catalog.remembered-model",
      JSON.stringify({ modelId: 7, at: 0 }),
    )
    expect(takeRememberedModel(0)).toBeNull()
  })
})
