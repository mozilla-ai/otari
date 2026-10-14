/**
 * A catalog model's offerings, for a control that names one by model and then
 * by provider rather than as a bare `provider:model` selector.
 */

import type { CatalogModelSummary } from "@/client"
import type { SelectOption } from "@/design-system/forms/Select"
import {
  isPrefixedSelector,
  providerInstanceOf,
  providerModelId,
} from "@/features/models/modelKey"
import { providerDisplayName } from "@/shared/helpers/providers"

/** The model among `models` that offers `selector`, if any. */
export function findOfferingOwner(
  models: readonly (CatalogModelSummary | undefined)[],
  selector: string,
): CatalogModelSummary | undefined {
  const wanted = selector.trim()
  if (wanted === "") return undefined
  return models.find((model) => model?.selectors.includes(wanted))
}

/** A model's offerings as provider choices, each saving its own selector. */
export function offeringChoices(model: CatalogModelSummary): SelectOption[] {
  const selectors = model.selectors.filter(isPrefixedSelector)
  const providers = selectors.map(
    (selector) => providerInstanceOf(selector) ?? "",
  )
  return selectors
    .map((selector, index) => {
      const provider = providers[index] ?? ""
      const label = providerDisplayName(provider)
      // Two builds on one instance would otherwise read as one choice twice.
      const isShared =
        providers.filter((other) => other === provider).length > 1
      return {
        value: selector,
        label: isShared ? `${label} (${providerModelId(selector)})` : label,
      }
    })
    .sort((a, b) => a.label.localeCompare(b.label))
}

/**
 * The offering a fresh pick of `model` saves: the one its catalog id resolves
 * to, which is the gateway's own best choice, else the first one listed.
 */
export function preferredSelector(model: CatalogModelSummary): string {
  const selectors = model.selectors.filter(isPrefixedSelector)
  const resolved = model.resolves_to ?? ""
  return selectors.includes(resolved) ? resolved : (selectors[0] ?? "")
}

/**
 * A selector in words: the model and the provider serving it.
 *
 * Without a catalog entry, the provider's own model id stands in for the
 * name. Text that is not a selector at all comes back as itself.
 */
export function describeOffering(
  selector: string,
  owner?: CatalogModelSummary,
): { model: string; provider: string } {
  const trimmed = selector.trim()
  if (!isPrefixedSelector(trimmed)) return { model: trimmed, provider: "" }
  return {
    model: owner?.name ?? providerModelId(trimmed),
    provider: providerDisplayName(providerInstanceOf(trimmed) ?? ""),
  }
}

/** "Claude Sonnet 5.5 on Anthropic", or the bare text for a non-selector. */
export function offeringPhrase(
  selector: string,
  owner?: CatalogModelSummary,
): string {
  const { model, provider } = describeOffering(selector, owner)
  return provider ? `${model} on ${provider}` : model
}
