/**
 * The form rules over a subagent router, and the draft the form edits.
 *
 * The wire shapes come from `@/client`; the rules here mirror the server's
 * (`services/routing/_agent_recommender_rules.py`), which applies them again.
 * There is no field for a stored key, because the API never returns one.
 */

import type {
  AgentRecommender,
  ConfiguredRecommender,
  CreateAgentRecommenderRequest,
  UpdateAgentRecommenderRequest,
} from "@/client"

export interface RecommenderDraft {
  provider: string
  model: string
  apiBase: string
  apiKey: string
  inputPrice: string
  outputPrice: string
}

export const EMPTY_DRAFT: RecommenderDraft = {
  provider: "typesafe",
  model: "",
  apiBase: "",
  apiKey: "",
  inputPrice: "",
  outputPrice: "",
}

export const DECISION_PROVIDERS: readonly { value: string; label: string }[] = [
  { value: "typesafe", label: "TypeSafe" },
  { value: "openrouter", label: "OpenRouter" },
  { value: "llamacpp", label: "llama.cpp server" },
]

/** A server the deployment runs itself: no default address, and normally no key. */
const SELF_HOSTED = new Set(["llamacpp"])

export function isSelfHosted(provider: string): boolean {
  return SELF_HOSTED.has(provider)
}

export type DraftErrors = Partial<
  Record<"model" | "apiBase" | "apiKey" | "inputPrice" | "outputPrice", string>
>

export type PriceParse =
  | { ok: true; value: number }
  | { ok: false; message: string }

/**
 * Six places, the ledger's precision: a finer price would be rounded where it
 * is charged. Twelve whole digits, more than the store keeps.
 */
const PRICE = /^\d{1,12}(\.\d{1,6})?$/

/** A price in USD per million tokens, zero when blank. */
export function parsePrice(text: string): PriceParse {
  const trimmed = text.trim()
  if (trimmed === "") return { ok: true, value: 0 }
  if (!PRICE.test(trimmed)) {
    return {
      ok: false,
      message:
        "Enter an amount in USD, such as 0.15, with at most six decimal places.",
    }
  }
  return { ok: true, value: Number(trimmed) }
}

export function providerLabel(provider: string): string {
  return (
    DECISION_PROVIDERS.find((option) => option.value === provider)?.label ??
    provider
  )
}

/** A stored rate as the form shows it: the shortest spelling of the number. */
function priceText(price: number): string {
  return String(price)
}

export function priceSummary(
  recommender: Pick<
    AgentRecommender,
    "input_price_per_million" | "output_price_per_million"
  >,
): string {
  const input = priceText(recommender.input_price_per_million)
  const output = priceText(recommender.output_price_per_million)
  return `$${input} in, $${output} out per 1M tokens`
}

/** A draft from a stored router, without its key. */
export function draftFrom(recommender: AgentRecommender): RecommenderDraft {
  return {
    provider: recommender.provider,
    model: recommender.model,
    apiBase: recommender.api_base ?? "",
    apiKey: "",
    inputPrice: priceText(recommender.input_price_per_million),
    outputPrice: priceText(recommender.output_price_per_million),
  }
}

/**
 * A draft started from the decision model config.yml names, so adding it here
 * with a price is one step. It carries no key and no price: those are what
 * the row adds.
 */
export function draftFromConfigured(
  configured: ConfiguredRecommender,
): RecommenderDraft {
  return {
    ...EMPTY_DRAFT,
    provider: configured.provider,
    model: configured.model,
  }
}

/** Whether saving without typing a key keeps one already stored: the key goes to the API base. */
function keepsStoredKey(
  draft: RecommenderDraft,
  stored: AgentRecommender | undefined,
): boolean {
  return (
    stored !== undefined &&
    stored.provider === draft.provider &&
    (stored.api_base ?? "") === draft.apiBase.trim() &&
    stored.has_api_key
  )
}

/** `stored` is the router being edited; absent for a new one. */
export function checkDraft(
  draft: RecommenderDraft,
  stored: AgentRecommender | undefined,
): DraftErrors {
  const errors: DraftErrors = {}
  if (draft.model.trim() === "") errors.model = "Enter the decision model."
  const hasKey = draft.apiKey.trim() !== "" || keepsStoredKey(draft, stored)
  const selfHosted = isSelfHosted(draft.provider)
  if (!hasKey && !selfHosted)
    errors.apiKey = "Enter the API key for this provider."
  const apiBase = draft.apiBase.trim()
  // The server reads a scheme in any case.
  const scheme = apiBase.toLowerCase()
  if (apiBase === "" && selfHosted)
    errors.apiBase = "Enter the server's address."
  else if (apiBase !== "" && !/^https?:\/\//.test(scheme))
    errors.apiBase = "Enter an http or https address."
  else if (hasKey && scheme.startsWith("http://"))
    errors.apiBase = "Use https: the API key is sent to this address."
  const prices = (["inputPrice", "outputPrice"] as const).map((name) => {
    const parsed = parsePrice(draft[name])
    if (!parsed.ok) errors[name] = parsed.message
    return parsed
  })
  // A stored router charges the caller, so it names a rate.
  if (prices.every((parsed) => parsed.ok && parsed.value === 0))
    errors.inputPrice = "Set an input or output token price above zero."
  return errors
}

/** A checked price as the API takes it: blank is zero. */
function priceValue(text: string): number {
  const parsed = parsePrice(text)
  return parsed.ok ? parsed.value : 0
}

export function createBody(
  draft: RecommenderDraft,
): CreateAgentRecommenderRequest {
  const apiBase = draft.apiBase.trim()
  const apiKey = draft.apiKey.trim()
  return {
    provider: draft.provider,
    model: draft.model.trim(),
    ...(apiBase === "" ? {} : { api_base: apiBase }),
    ...(apiKey === "" ? {} : { api_key: apiKey }),
    input_price_per_million: priceValue(draft.inputPrice),
    output_price_per_million: priceValue(draft.outputPrice),
  }
}

/** The PATCH always carries `api_base`: blank goes back to the provider's own address. */
export function patchBody(
  draft: RecommenderDraft,
): UpdateAgentRecommenderRequest {
  const { api_base, ...body } = createBody(draft)
  return { ...body, api_base: api_base ?? "" }
}

/** What the key box says when a stored key would be kept; blank otherwise. */
export function keyPlaceholder(
  draft: RecommenderDraft,
  stored: AgentRecommender | undefined,
): string {
  if (!keepsStoredKey(draft, stored)) return ""
  return stored?.api_key_last4
    ? `Stored key ends in ${stored.api_key_last4}. Leave blank to keep it.`
    : "A key is stored. Leave blank to keep it."
}
