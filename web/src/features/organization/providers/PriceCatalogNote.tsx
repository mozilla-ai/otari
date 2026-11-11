import type { OrgProviderPriceCatalog } from "@/client"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { formatDate } from "@/shared/helpers/format"

// Which models.dev prices the default rates follow, said beside Refresh pricing
// because that button moves a rate onto those prices and no further. Newer
// prices the deployment has fetched are named too: without that, pressing
// Refresh pricing and seeing nothing change reads as a broken button, when the
// prices it needs are waiting for an operator's catalog update.

/** The sentence naming the prices in force. */
export function pricesInForce(catalog: OrgProviderPriceCatalog): string {
  const date = formatDate(catalog.prices_from)
  return catalog.bundled
    ? `Default prices come from the models.dev catalog packaged with Otari, made ${date}.`
    : `Default prices come from the models.dev catalog of ${date}.`
}

export function PriceCatalogNote({
  catalog,
}: {
  catalog: OrgProviderPriceCatalog
}) {
  if (!catalog.newer_at) {
    return <p className="text-caption">{pricesInForce(catalog)}</p>
  }
  return (
    <InfoBanner>
      <p>
        {pricesInForce(catalog)} Newer prices, fetched{" "}
        {formatDate(catalog.newer_at)}, are not in force yet.
      </p>
      <p className="mt-1">
        A deployment operator must update the catalog before Refresh pricing can
        use them.
      </p>
    </InfoBanner>
  )
}
