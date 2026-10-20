/**
 * The form rules of a subagent router, kept apart from the rendering so they
 * can be tested without mounting the dialog. They mirror the server's, which
 * applies them again (`services/routing/_recommender_rules.py`).
 *
 * Prices are held as the strings the boxes contain, in USD per million of the
 * decision model's tokens. There is no field here for a stored key, because
 * the API never returns one.
 */

import type {
  AgentRecommender,
  CreateAgentRecommenderRequest,
  DecisionProviderOption,
  UpdateAgentRecommenderRequest,
} from "@/client"
import { formatUnitRate } from "@/shared/helpers/format"

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

/** What each decision provider the server lists is called here; one it adds later shows by its own name. */
const PROVIDER_LABELS: Record<string, string> = {
  typesafe: "TypeSafe",
  openrouter: "OpenRouter",
  llamacpp: "llama.cpp server",
}

export function providerLabel(provider: string): string {
  return PROVIDER_LABELS[provider] ?? provider
}

export function providerOptions(
  providers: readonly DecisionProviderOption[],
): { value: string; label: string }[] {
  return providers.map(({ provider }) => ({
    value: provider,
    label: providerLabel(provider),
  }))
}

/** A server the deployment runs itself: no default address, and normally no key. */
export function isSelfHosted(
  provider: string,
  providers: readonly DecisionProviderOption[],
): boolean {
  return providers.some(
    (option) => option.provider === provider && option.self_hosted,
  )
}

export type DraftErrors = Partial<
  Record<"model" | "apiBase" | "apiKey" | "inputPrice" | "outputPrice", string>
>

export type PriceParse =
  | { ok: true; value: number }
  | { ok: false; message: string }

/**
 * Six places, the ledger's precision: a finer price would be rounded where it
 * is charged. Twelve whole digits, what the store keeps.
 */
const PRICE = /^\d{1,12}(\.\d{1,6})?$/

/** A price in USD, zero when blank. */
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

export function priceSummary(recommender: AgentRecommender): string {
  const input = formatUnitRate(recommender.input_price_per_million)
  const output = formatUnitRate(recommender.output_price_per_million)
  return `${input} in, ${output} out per 1M tokens`
}

/** A draft from a stored recommender, without its key. */
export function draftFrom(recommender: AgentRecommender): RecommenderDraft {
  return {
    provider: recommender.provider,
    model: recommender.model,
    apiBase: recommender.api_base ?? "",
    apiKey: "",
    inputPrice: String(recommender.input_price_per_million),
    outputPrice: String(recommender.output_price_per_million),
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

/** `stored` is the recommender being edited, absent for a new one. */
export function checkDraft(
  draft: RecommenderDraft,
  stored: AgentRecommender | undefined,
  providers: readonly DecisionProviderOption[],
): DraftErrors {
  const errors: DraftErrors = {}
  if (draft.model.trim() === "") errors.model = "Enter the decision model."
  const hasKey = draft.apiKey.trim() !== "" || keepsStoredKey(draft, stored)
  const selfHosted = isSelfHosted(draft.provider, providers)
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
  // The price on the row is what a recommendation is charged.
  if (prices.every((parsed) => parsed.ok && parsed.value === 0))
    errors.inputPrice = "Set an input or output token price above zero."
  return errors
}

/** A checked price as the API takes it: blank is zero. */
function priceValue(text: string): number {
  const parsed = parsePrice(text)
  return parsed.ok ? parsed.value : 0
}

/** A new recommender, landing switched off: the form switches it on once everything it saves has landed. */
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
    enabled: false,
  }
}

/**
 * A change to a stored recommender, switching it off until the form switches
 * it back on. A blank `api_base` goes back to the provider's own; an absent
 * key keeps the stored one.
 */
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
