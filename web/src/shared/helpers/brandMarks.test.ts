import { describe, expect, it } from "vitest"
import {
  hasProviderMark,
  isProductMarkProvider,
  makerMark,
  providerMark,
} from "@/shared/helpers/brandMarks"

describe("providerMark", () => {
  it("resolves a provider id to geometry", () => {
    const mark = providerMark("mistral")

    expect(mark).toBeDefined()
    expect(mark?.viewBox).toBe("0 0 24 24")
    expect(mark?.shapes.length).toBeGreaterThan(0)
    expect(mark?.shapes[0]?.d).toMatch(/^[Mm]/)
  })

  it("matches case-insensitively on a trimmed id", () => {
    // An id reaches the dashboard from a catalog row, a stored credential and a
    // model selector's prefix, and those do not agree on case or padding.
    const mark = providerMark("mistral")

    expect(providerMark(" Mistral ")).toEqual(mark)
    expect(providerMark("MISTRAL")).toEqual(mark)
  })

  it("returns undefined for an id it has no mark for", () => {
    // The ordinary case, not a fault: an operator names their own instances.
    expect(providerMark("my-local-llm")).toBeUndefined()
    expect(providerMark("")).toBeUndefined()
    // Both synthetic provider types. An API dialect is not a vendor, so a mark
    // would assert a relationship that does not exist.
    expect(providerMark("openai-compatible")).toBeUndefined()
    expect(providerMark("anthropic-compatible")).toBeUndefined()
  })

  it("gives the sibling platform ids the same mark", () => {
    // The mark names the platform vendor the credential belongs to and the name
    // carries the rest, so sharing one is intended rather than a collision.
    const azure = providerMark("azure")

    expect(providerMark("azureopenai")).toEqual(azure)
    expect(providerMark("azureanthropic")).toEqual(azure)
    expect(providerMark("vertexaianthropic")).toEqual(providerMark("vertexai"))
    expect(providerMark("llama")).toEqual(providerMark("meta"))
  })

  it("tiles both AWS provider rows rather than showing the wordmark", () => {
    // The rule would give them the AWS mark, but the only one in the set is the
    // lowercase wordmark, unreadable at 16px. Asserted as absence, which is what
    // makes the component draw a lettermark instead.
    expect(providerMark("bedrock")).toBeUndefined()
    expect(providerMark("sagemaker")).toBeUndefined()
    expect(hasProviderMark("bedrock")).toBe(false)
    // And the maker row for the same company agrees, so a card and a rail row
    // never disagree about Amazon.
    expect(makerMark("amazon")).toBeUndefined()
  })

  it("ships LM Studio as a single tone", () => {
    // Its lighter tone measures 2.67:1 on the dark ground and 2.00:1 on the
    // light one, under the 3:1 floor for a non-text graphic, so it is flattened
    // at the source rather than rendered and hoped for.
    const mark = providerMark("lmstudio")

    expect(mark?.shapes.length).toBeGreaterThan(1)
    expect(mark?.shapes.every((shape) => shape.fillOpacity === undefined)).toBe(
      true,
    )
  })

  it("keeps the Azure tones, which clear the floor", () => {
    // 5.36:1 and 3.55:1 for the lightest of them, so this one is not flattened.
    // Paired with the LM Studio case: together they pin the rule rather than the
    // outcome for one mark.
    const tones = providerMark("azure")?.shapes.map(
      (shape) => shape.fillOpacity,
    )

    expect(tones).toContain(0.5)
    expect(tones).toContain(0.75)
  })

  it("carries no color of its own", () => {
    // Marks inherit the page's text ink. Any fill baked into the data would
    // survive a retheme and break in one of the two grounds.
    const withFill = Object.values([
      "openai",
      "anthropic",
      "mistral",
      "azure",
      "lmstudio",
      "llamacpp",
    ]).filter((id) =>
      providerMark(id)?.shapes.some((shape) => /#|rgb|hsl/.test(shape.d)),
    )

    expect(withFill).toEqual([])
  })
})

describe("hasProviderMark", () => {
  it("counts a glyph and the product mark alike", () => {
    // The slot decision needs "is there a mark", and ours has no geometry in
    // this module, so asking `providerMark` alone would under-count it.
    expect(hasProviderMark("mistral")).toBe(true)
    expect(hasProviderMark("otari")).toBe(true)
    expect(hasProviderMark("mzai")).toBe(true)
    expect(hasProviderMark("my-local-llm")).toBe(false)
  })

  it("keeps our own ids out of the geometry table", () => {
    // Deliberate: the Otari path is written down once, in the design system.
    expect(providerMark("otari")).toBeUndefined()
    expect(isProductMarkProvider("Otari")).toBe(true)
    expect(isProductMarkProvider("openai")).toBe(false)
  })
})

describe("makerMark", () => {
  it("keys on the vendor slug, which is not the provider id", () => {
    // The three companies whose two keys disagree. A single table would have
    // missed all of them, which is why there are two.
    expect(makerMark("mistralai")).toBeDefined()
    expect(makerMark("z-ai")).toBeDefined()
    expect(makerMark("moonshotai")).toBeDefined()
    // The provider ids for the same companies resolve in the other table only.
    expect(makerMark("mistral")).toBeUndefined()
    expect(makerMark("zai")).toBeUndefined()
    expect(makerMark("moonshot")).toBeUndefined()
  })

  it("shares the glyph with the provider of the same company", () => {
    expect(makerMark("mistralai")).toEqual(providerMark("mistral"))
    expect(makerMark("openai")).toEqual(providerMark("openai"))
  })

  it("gives Google its own mark, not the Gemini sparkle", () => {
    // `gemini` is the provider and wears the sparkle; the maker is Google and
    // wears the G. They are different marks on purpose.
    expect(makerMark("google")).toBeDefined()
    expect(makerMark("google")).not.toEqual(providerMark("gemini"))
  })

  it("tiles the two makers that have no usable mark", () => {
    // Amazon's only mark in the set is the aws wordmark, unreadable at these
    // steps, so it tiles like the Bedrock and SageMaker rows. OpenBMB has none.
    expect(makerMark("amazon")).toBeUndefined()
    expect(makerMark("openbmb")).toBeUndefined()
  })

  it("covers every maker the gateway can report", () => {
    // The vocabulary is closed, so this is the whole set. A maker added to
    // `model_identity.py` without a mark shows up here rather than as a silent
    // tile on the page.
    const MAKER_SLUGS = [
      "ai21",
      "alibaba",
      "amazon",
      "anthropic",
      "bytedance",
      "cohere",
      "deepseek",
      "google",
      "meta",
      "microsoft",
      "minimax",
      "mistralai",
      "moonshotai",
      "nousresearch",
      "nvidia",
      "openai",
      "openbmb",
      "perplexity",
      "xai",
      "z-ai",
    ]
    const TILED = ["amazon", "openbmb"]

    expect(MAKER_SLUGS.filter((slug) => makerMark(slug) === undefined)).toEqual(
      TILED,
    )
  })
})
