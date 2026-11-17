import { describe, expect, it } from "vitest"

import {
  publicProvidersHref,
  publicProvidersPath,
} from "@/features/providers/publicProviders"

describe("publicProvidersPath", () => {
  it("names the list and a selected provider", () => {
    expect(publicProvidersPath("/providers")).toEqual({ name: undefined })
    expect(publicProvidersPath("/providers/")).toEqual({ name: undefined })
    expect(publicProvidersPath("/providers/fireworks")).toEqual({
      name: "fireworks",
    })
    expect(publicProvidersPath("/providers/openai-eu/")).toEqual({
      name: "openai-eu",
    })
  })

  it("ignores a query string and refuses everything else", () => {
    expect(publicProvidersPath("/providers?x=1")).toEqual({ name: undefined })
    expect(publicProvidersPath("/providers/fireworks?x=1")).toEqual({
      name: "fireworks",
    })
    expect(publicProvidersPath("/")).toBeNull()
    expect(publicProvidersPath("/models")).toBeNull()
    expect(publicProvidersPath("/providers-x")).toBeNull()
    expect(publicProvidersPath("/providers/a/b")).toBeNull()
    expect(publicProvidersPath("/providers/%")).toBeNull()
  })

  it("round-trips through the href", () => {
    expect(publicProvidersPath(publicProvidersHref("openai eu"))).toEqual({
      name: "openai eu",
    })
    expect(publicProvidersHref()).toBe("/providers")
  })
})
