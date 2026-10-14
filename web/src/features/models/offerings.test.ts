import { describe, expect, it } from "vitest"

import { catalogModelSummary } from "@/tests/fixtures"

import {
  describeOffering,
  findOfferingOwner,
  offeringChoices,
  offeringPhrase,
  preferredSelector,
} from "./offerings"

const SONNET = catalogModelSummary({
  id: "anthropic/claude-sonnet-5.5",
  name: "Claude Sonnet 5.5",
  selectors: [
    "anthropic:claude-sonnet-5-5",
    "bedrock:anthropic.claude-sonnet-5-5-v1:0",
  ],
  providers: ["anthropic", "bedrock"],
  resolves_to: "bedrock:anthropic.claude-sonnet-5-5-v1:0",
})

describe("offerings", () => {
  it("finds the model that offers a selector, and nothing for other text", () => {
    expect(
      findOfferingOwner([undefined, SONNET], "anthropic:claude-sonnet-5-5"),
    ).toBe(SONNET)
    expect(findOfferingOwner([SONNET], "claude")).toBeUndefined()
    expect(findOfferingOwner([SONNET], "")).toBeUndefined()
  })

  it("offers each provider by name, saving its own selector", () => {
    expect(offeringChoices(SONNET)).toEqual([
      {
        value: "bedrock:anthropic.claude-sonnet-5-5-v1:0",
        label: "Amazon Bedrock",
      },
      { value: "anthropic:claude-sonnet-5-5", label: "Anthropic" },
    ])
  })

  it("tells two builds on one provider apart by their model id", () => {
    const model = catalogModelSummary({
      id: "openai/gpt-5",
      selectors: ["openai:gpt-5", "openai:gpt-5-2026-08-01"],
    })
    expect(
      offeringChoices(model)
        .map((choice) => choice.label)
        .sort(),
    ).toEqual(["OpenAI (gpt-5)", "OpenAI (gpt-5-2026-08-01)"].sort())
  })

  it("prefers the offering the catalog id resolves to", () => {
    expect(preferredSelector(SONNET)).toBe(
      "bedrock:anthropic.claude-sonnet-5-5-v1:0",
    )
    expect(
      preferredSelector({ ...SONNET, resolves_to: "elsewhere:model" }),
    ).toBe("anthropic:claude-sonnet-5-5")
  })

  it("names a selector by its model and provider, with or without the catalog", () => {
    expect(describeOffering("anthropic:claude-sonnet-5-5", SONNET)).toEqual({
      model: "Claude Sonnet 5.5",
      provider: "Anthropic",
    })
    expect(describeOffering("openai:gpt-5-mini")).toEqual({
      model: "gpt-5-mini",
      provider: "OpenAI",
    })
    expect(offeringPhrase("anthropic:claude-sonnet-5-5", SONNET)).toBe(
      "Claude Sonnet 5.5 on Anthropic",
    )
    // Text that is not a selector stays as typed.
    expect(offeringPhrase("claude")).toBe("claude")
  })
})
