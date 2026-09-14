import { describe, expect, it } from "vitest"
import { providerDisplayName } from "@/shared/helpers/providers"

describe("providerDisplayName", () => {
  it("uses each vendor's own capitalization", () => {
    expect(providerDisplayName("openai")).toBe("OpenAI")
    expect(providerDisplayName("mistral")).toBe("Mistral AI")
    expect(providerDisplayName("xai")).toBe("xAI")
    expect(providerDisplayName("llamacpp")).toBe("llama.cpp")
  })

  it("never title-cases an id it does not know", () => {
    // The failure this map exists to prevent: a generic rule turns `openai`
    // into "Openai". An unknown id is left exactly as it arrived instead, so a
    // provider added upstream reads as plain rather than as misspelled.
    expect(providerDisplayName("some-new-backend")).toBe("some-new-backend")
    expect(providerDisplayName("")).toBe("")
  })

  it("matches whatever case the id arrives in", () => {
    expect(providerDisplayName("OpenAI")).toBe("OpenAI")
    expect(providerDisplayName(" Anthropic ")).toBe("Anthropic")
  })
})
