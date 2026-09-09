import { describe, expect, it } from "vitest"

import {
  publicCatalogHref,
  publicCatalogPath,
} from "@/features/models/publicCatalog"

describe("publicCatalogPath", () => {
  it("names the list and a selected model", () => {
    expect(publicCatalogPath("#/models")).toEqual({ modelId: undefined })
    expect(publicCatalogPath("#/models/")).toEqual({ modelId: undefined })
    expect(publicCatalogPath("#/models/glm-5-3")).toEqual({
      modelId: "glm-5-3",
    })
  })

  it("ignores a query string and refuses everything else", () => {
    expect(publicCatalogPath("#/models?provider=nebius")).toEqual({
      modelId: undefined,
    })
    expect(publicCatalogPath("#/")).toBeNull()
    expect(publicCatalogPath("#/keys")).toBeNull()
    expect(publicCatalogPath("#/models/a/b")).toBeNull()
  })

  it("round-trips through the href", () => {
    expect(publicCatalogPath(publicCatalogHref("kimi-k2-6"))).toEqual({
      modelId: "kimi-k2-6",
    })
    expect(publicCatalogHref()).toBe("#/models")
  })
})
