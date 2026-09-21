/**
 * Which companies have a mark, without the geometry that draws one.
 *
 * Split from {@link brandMarks} because the two questions have different
 * costs. "Does this list need a mark slot" is asked at first paint, for every
 * row, and answering it needs ~60 short strings. "What shapes draw this mark"
 * is 70 kB of path data that only matters once something is on screen. Held
 * together, every visitor to the Models page downloaded all of the art before
 * the page painted, for the handful of providers a deployment actually has.
 *
 * So this module stays a static import and the geometry is fetched on demand.
 * The cost of the split is that these lists can drift from the tables they
 * mirror, which is what `brandMarkKeys.test.ts` exists to prevent: it compares
 * them against the tables and fails on any difference in either direction.
 */

/** Provider ids and instance names with a mark. */
const PROVIDER_MARK_IDS: ReadonlySet<string> = new Set([
  "anthropic",
  "atlascloud",
  "azure",
  "azureanthropic",
  "azureopenai",
  "cerebras",
  "cohere",
  "dashscope",
  "databricks",
  "deepinfra",
  "deepseek",
  "fireworks",
  "gemini",
  "github",
  "gmi",
  "groq",
  "huggingface",
  "inception",
  "llama",
  "llamacpp",
  "lmstudio",
  "meta",
  "minimax",
  "mistral",
  "moonshot",
  "nebius",
  "ollama",
  "openai",
  "openrouter",
  "ovhcloud",
  "perplexity",
  "qiniu",
  "sambanova",
  "together",
  "vertexai",
  "vertexaianthropic",
  "vllm",
  "voyage",
  "watsonx",
  "xai",
  "zai",
])

/** Vendor slugs with a maker mark. */
const MAKER_MARK_SLUGS: ReadonlySet<string> = new Set([
  "ai21",
  "alibaba",
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
  "perplexity",
  "xai",
  "z-ai",
])

/**
 * The ids this deployment serves its own models under.
 *
 * They take the Otari mark, which lives in `ProductMark` rather than in the
 * geometry table: it is already written down once in the design system, and a
 * second copy would be a second place to edit when the mark moves.
 *
 * They are named for the reason every vendor is. A deployment serving its own
 * models would otherwise show a lettermark tile while every third party got a
 * logo, which is the bug this pair of modules exists to fix, pointed at us.
 */
const PRODUCT_MARK_IDS: readonly string[] = ["mzai", "otari"]

const normalize = (value: string): string => value.trim().toLowerCase()

/** Whether this id is one of ours, and so takes the Otari mark. */
export function isProductMarkProvider(providerId: string): boolean {
  return PRODUCT_MARK_IDS.includes(normalize(providerId))
}

/**
 * Whether a provider has any mark at all, geometry or the product mark.
 *
 * A `Set` rather than an object index, so an operator-chosen instance named
 * `constructor` or `toString` cannot resolve a key off a prototype.
 */
export function hasProviderMark(providerId: string): boolean {
  return (
    isProductMarkProvider(providerId) ||
    PROVIDER_MARK_IDS.has(normalize(providerId))
  )
}

/** The same question for a model maker, by its vendor slug. */
export function hasMakerMark(vendorSlug: string): boolean {
  return MAKER_MARK_SLUGS.has(normalize(vendorSlug))
}

/** Read by the drift gate beside this module, not by the app. */
export const MARK_KEYS_FOR_TEST = {
  providers: PROVIDER_MARK_IDS,
  makers: MAKER_MARK_SLUGS,
}
