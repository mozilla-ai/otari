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
 * (`isValidModelKey`, in `SetPriceDialog` and in `organization/pricingOverride`),
 * including the legacy slash form the gateway collapses onto the colon one.
 * Stated here rather than borrowed from either, because this asks a question
 * about a catalog entry and those ask one about a draft, and a control that
 * imported a dialog would close a cycle with the dialog that renders it.
 */

const PREFIXED_SELECTOR = /^([^\s:/]+)[:/][^\s]+$/

/** Whether a catalog id names a model a provider serves, rather than standing for one. */
export function isPrefixedSelector(modelId: string): boolean {
  return PREFIXED_SELECTOR.test(modelId)
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
