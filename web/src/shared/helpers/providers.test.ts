import { describe, expect, it } from "vitest"
import {
  providerCatalogName,
  providerDisplayName,
} from "@/shared/helpers/providers"

describe("providerDisplayName", () => {
  it("uses each vendor's own capitalization", () => {
    expect(providerDisplayName("openai")).toBe("OpenAI")
    expect(providerDisplayName("mistral")).toBe("Mistral AI")
    expect(providerDisplayName("xai")).toBe("xAI")
    expect(providerDisplayName("llamacpp")).toBe("llama.cpp")
    // The wire id is `together`, not `togetherai`: this map's keys come from
    // `AnyLLM.get_supported_providers()`, and a key that is not a real id is a
    // row that silently never gets its name.
    expect(providerDisplayName("together")).toBe("Together AI")
    // Our own two, held to the same rule: a deployment serving its own models
    // would otherwise read lowercase beside correctly cased third parties.
    expect(providerDisplayName("mzai")).toBe("Mozilla AI")
    expect(providerDisplayName("otari")).toBe("Otari")
    // Two different any-llm providers, so two different names: given one label
    // they produced two identical options in the Models provider filter.
    expect(providerDisplayName("azure")).not.toBe(
      providerDisplayName("azureopenai"),
    )
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

describe("providerCatalogName", () => {
  // The rows the backend sends for the add-provider picker, which name two ids
  // alike ("Azure", "Vertex") and fall back to the bare id for five more.
  const rows = [
    ["azure", "Azure"],
    ["azureopenai", "Azure"],
    ["vertexai", "Vertex"],
    ["vertexaianthropic", "Vertex"],
    ["mistral", "Mistral"],
    ["gemini", "Google"],
    ["lmstudio", "LMStudio"],
    ["minimax", "MiniMax (minimax.io)"],
    ["cascadia", "cascadia"],
    ["liner", "liner"],
    ["neosantara", "neosantara"],
    ["tokenhub", "tokenhub"],
    ["typesafe", "typesafe"],
  ] as const

  it("gives every picker row a distinct name", () => {
    const names = rows.map(([id, name]) => providerCatalogName(id, name))
    expect(new Set(names).size).toBe(rows.length)
  })

  it("prefers our name over the backend's, then the backend's, then the id", () => {
    expect(providerCatalogName("azure", "Azure")).toBe("Azure AI Foundry")
    expect(providerCatalogName("azureopenai", "Azure")).toBe("Azure OpenAI")
    expect(providerCatalogName("vertexaianthropic", "Vertex")).toBe(
      "Vertex AI Anthropic",
    )
    expect(providerCatalogName("typesafe", "typesafe")).toBe("TypeSafe")
    expect(providerCatalogName(" AZURE ", "Azure")).toBe("Azure AI Foundry")
    expect(providerCatalogName("brand-new", "Brand New")).toBe("Brand New")
    expect(providerCatalogName("brand-new", undefined)).toBe("brand-new")
  })
})
