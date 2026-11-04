import type {
  HostedCatalogRefresh,
  HostedCatalogUpdatePreview,
  HostedModelsRefresh,
} from "@/client"

const plural = (count: number, one: string, many: string) =>
  `${count} ${count === 1 ? one : many}`

type HostedProviderReport = HostedCatalogRefresh["providers"][number]

/** The providers whose model lists outlived whatever defined them, such as a removed `config.yml` entry. */
function orphanedRosters(result: HostedCatalogRefresh): string | undefined {
  const orphaned = result.orphaned_rosters ?? []
  if (orphaned.length === 0) return undefined
  return orphaned.join(", ")
}

/**
 * One sentence for what a refresh did. A refusal comes back in the body rather
 * than thrown, so it is worded here too.
 */
export function refreshOutcome(result: HostedModelsRefresh): string {
  if (result.error) return result.error
  const parts = [
    result.added.length > 0
      ? `Offered ${plural(result.added.length, "new model", "new models")}.`
      : undefined,
    result.repriced.length > 0
      ? `Moved ${plural(result.repriced.length, "default price", "default prices")} to today's rate.`
      : undefined,
  ].filter((part) => part !== undefined)
  return parts.length > 0 ? parts.join(" ") : "The provider lists nothing new."
}

/**
 * One sentence for what a deployment-wide sweep did.
 *
 * Counts across every provider, because that is what the operator pressed one
 * button for. A provider that would not answer is not folded in here: it gets
 * its own line (`catalogRefreshFailures`), so a sweep that offered nothing
 * because one upstream was down does not read as a sweep that found nothing.
 */
export function catalogRefreshOutcome(result: HostedCatalogRefresh): string {
  const total = (pick: (provider: HostedProviderReport) => string[]) =>
    result.providers.reduce((sum, provider) => sum + pick(provider).length, 0)
  const added = total((provider) => provider.added)
  const repriced = total((provider) => provider.repriced ?? [])
  const orphaned = orphanedRosters(result)
  const parts = [
    added > 0
      ? `Offered ${plural(added, "new model", "new models")}.`
      : undefined,
    repriced > 0
      ? `Moved ${plural(repriced, "default price", "default prices")} to today's rate.`
      : undefined,
    result.removed_count > 0
      ? `Removed ${plural(result.removed_count, "model", "models")} nothing here serves.`
      : undefined,
    orphaned === undefined
      ? undefined
      : `Dropped the model lists of providers nothing configures any more: ${orphaned}.`,
  ].filter((part) => part !== undefined)
  return parts.length > 0
    ? parts.join(" ")
    : "Nothing to change: the catalog already matches this page."
}

/**
 * One sentence for what a sweep is about to do, for the confirm dialog.
 *
 * Future tense and counts only: the models themselves are listed beside this,
 * because naming them is the whole reason the preview exists.
 */
export function catalogPreviewOutcome(result: HostedCatalogRefresh): string {
  const orphaned = orphanedRosters(result)
  const lists =
    orphaned === undefined
      ? undefined
      : `The model lists of providers nothing configures any more will be dropped: ${orphaned}.`
  if (result.removed_count === 0) {
    return (
      lists ?? "Nothing to remove: every priced model has something serving it."
    )
  }
  const rates = plural(result.removed_price_rows, "stored rate", "stored rates")
  const overrides = plural(
    result.removed_override_rows,
    "organization override",
    "organization overrides",
  )
  const them = result.removed_count === 1 ? "it" : "them"
  const models =
    `${plural(result.removed_count, "model", "models")} will be removed, ` +
    `taking ${rates} and ${overrides} with ${them}.`
  return lists === undefined ? models : `${models} ${lists}`
}

/**
 * One sentence for the models.dev half of a catalog update, for its confirm
 * dialog. A fetch that failed is said rather than thrown, because the model
 * lists are still updated without it.
 */
export function pricePreviewOutcome(
  preview: HostedCatalogUpdatePreview,
): string {
  if (preview.price_error) return preview.price_error
  const prices = preview.prices
  if (!prices) return "models.dev prices are unchanged."
  const parts = [
    prices.added_count > 0 ? `${prices.added_count} new` : undefined,
    prices.changed_count > 0 ? `${prices.changed_count} changed` : undefined,
    prices.removed_count > 0 ? `${prices.removed_count} removed` : undefined,
  ].filter((part) => part !== undefined)
  const total = prices.added_count + prices.changed_count + prices.removed_count
  return `models.dev has ${parts.join(", ")} ${total === 1 ? "price" : "prices"} to apply.`
}

/** One sentence for the models a catalog update would newly offer, across every provider. */
export function modelsPreviewOutcome(result: HostedCatalogRefresh): string {
  const added = result.providers.filter((provider) => provider.added.length > 0)
  const count = added.reduce((sum, provider) => sum + provider.added.length, 0)
  if (count === 0) return "No provider lists a model it does not already offer."
  return `${plural(count, "new model", "new models")} to offer on ${added
    .map((provider) => provider.provider)
    .join(", ")}, each served once something prices it.`
}

/** A provider a sweep could not read, and whether it is worth removing. */
export interface CatalogFailure {
  provider: string
  message: string
  /**
   * Whether the page should offer to remove this provider.
   *
   * True when the upstream answered the dial with an error, which leaves the
   * provider serving nothing until someone fixes or drops it. False where
   * deleting would be the wrong advice: a provider with no model listing serves
   * requests perfectly well and simply cannot be asked what it offers, and a
   * key this process cannot decrypt wants re-entering rather than destroying.
   * Never for a provider defined in configuration, which no delete reaches;
   * the sweep says which it is, so the answer does not hang on the list page.
   */
  offerRemoval: boolean
}

/** The providers a sweep could not read, one line each. */
export function catalogRefreshFailures(
  result: HostedCatalogRefresh,
): CatalogFailure[] {
  return result.providers
    .filter((provider) => provider.error || provider.discovery_unsupported)
    .map((provider) => ({
      provider: provider.provider,
      message: provider.error
        ? `${provider.provider}: ${provider.error}`
        : `${provider.provider} cannot list its models, so nothing new was offered.`,
      offerRemoval:
        Boolean(provider.error) &&
        !provider.credential_unreadable &&
        provider.source !== "config",
    }))
}
