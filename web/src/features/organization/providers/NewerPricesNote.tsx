import type { OrgProviderPriceCatalog } from "@/client"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { formatDate } from "@/shared/helpers/format"

// Newer models.dev prices the deployment has fetched and not applied. Said
// beside Refresh pricing because that button moves a rate onto the prices in
// force and no further: without this, pressing it and seeing nothing change
// reads as a broken button, when the prices it needs wait for a catalog update.
export function NewerPricesNote({
  catalog,
  canUpdate,
}: {
  catalog: OrgProviderPriceCatalog
  /** Whether the caller can run the catalog update from this panel. */
  canUpdate: boolean
}) {
  if (!catalog.newer_at) return null
  return (
    <InfoBanner>
      <p>
        Newer models.dev prices, fetched {formatDate(catalog.newer_at)}, are not
        in force yet.{" "}
        {canUpdate
          ? "Update catalog applies them."
          : "A deployment operator must update the catalog before Refresh pricing can use them."}
      </p>
    </InfoBanner>
  )
}
