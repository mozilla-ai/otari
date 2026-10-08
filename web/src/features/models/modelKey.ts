/**
 * What a `provider:model` selector is, for the controls that offer one.
 *
 * `GET /v1/models` lists aliases and routing policies alongside real models,
 * under a bare display name rather than a selector. Neither is an answer to
 * "which model is this rate stored under" or "which provider instance is this
 * ceiling narrowed to": the first stores a rate nothing reads back, the second
 * a cap that binds nothing.
 *
 * The same shape the two price forms validate what was typed against
 * (`isValidModelKey`), including the legacy slash form the gateway collapses
 * onto the colon one.
 */

const PREFIXED_SELECTOR = /^([^\s:/]+)[:/][^\s]+$/

/** Whether a catalog id names a model a provider serves, rather than standing for one. */
export function isPrefixedSelector(modelId: string): boolean {
  return PREFIXED_SELECTOR.test(modelId)
}

/**
 * Whether a typed price key could ever be read back.
 *
 * A pricing row is only resolved under a `prefix:model` selector (see
 * `normalize_pricing_key` in `services/provider_kwargs.py`), so a key with no
 * provider or instance prefix would store a rate nothing bills against.
 */
export function isValidModelKey(value: string): boolean {
  return isPrefixedSelector(value.trim())
}

/**
 * The provider instance a catalog id resolves through, or undefined for a name
 * that resolves through none.
 *
 * The prefix is what a budget scope is matched on
 * (`BudgetScopeRequest.provider_instance`), so it is the instance a ceiling
 * narrows to rather than the provider type behind it.
 */
export function providerInstanceOf(modelId: string): string | undefined {
  return PREFIXED_SELECTOR.exec(modelId)?.[1]
}

/** The provider's own model ID, from an `instance:model` selector. */
export function providerModelId(selector: string): string {
  return selector.slice(selector.indexOf(":") + 1)
}
