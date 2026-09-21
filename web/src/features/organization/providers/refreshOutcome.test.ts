import { describe, expect, it } from "vitest"

import type { OrgProviderModelsRefresh } from "@/client"

import { refreshOutcome } from "./refreshOutcome"

function result(
  over: Partial<OrgProviderModelsRefresh> = {},
): OrgProviderModelsRefresh {
  return {
    added: [],
    repriced: [],
    count: 0,
    error: null,
    discovery_unsupported: false,
    ...over,
  }
}

describe("refreshOutcome", () => {
  it("counts what was offered and what was repriced", () => {
    expect(
      refreshOutcome(
        result({ added: ["gpt-4o"], repriced: ["a", "b"], count: 3 }),
        "models",
      ),
    ).toBe("Offered 1 new model. Moved 2 default rates to today's.")
  })

  it("says which question had no news, per button", () => {
    expect(refreshOutcome(result(), "models")).toBe(
      "The provider lists nothing new.",
    )
    expect(refreshOutcome(result(), "pricing")).toBe(
      "Every default rate is already current.",
    )
  })

  it("surfaces a refusal verbatim, because it arrives in the body", () => {
    expect(
      refreshOutcome(
        result({ error: "the upstream refused the key" }),
        "models",
      ),
    ).toBe("the upstream refused the key")
  })

  it("tells an operator what to do when a provider publishes no list", () => {
    expect(
      refreshOutcome(result({ discovery_unsupported: true }), "models"),
    ).toBe(
      "This provider does not publish a model list. Add the models you want by name.",
    )
  })
})
