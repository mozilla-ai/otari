/**
 * What the deployment does with a model nobody has priced.
 *
 * Operator-only, because it reads `GET /v1/settings`, which is
 * `require_deployment_operator`. Its own file rather than an inner component of
 * the page that mounts it: two pages now show it, and a band whose whole content
 * is a sentence about deployment configuration is not a detail of either.
 */

import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { useSettings } from "@/shared/api/settings"

export function CatalogPolicy() {
  const settings = useSettings()
  if (settings.isLoading) return <PageLoading label="Loading pricing policy…" />
  if (settings.error) return <ErrorBanner error={settings.error} />
  if (!settings.data) return null

  if (settings.data.default_pricing) {
    return (
      <InfoBanner>
        Default pricing is on: a model with no stored price is metered at the
        upstream default below, so the table is the models you have overridden.
      </InfoBanner>
    )
  }
  return (
    <InfoBanner tone="warning">
      Default pricing is off, so the table below is everything this gateway can
      bill.{" "}
      {settings.data.require_pricing
        ? "A request for any other model is refused with HTTP 402, because require_pricing is on."
        : "A request for any other model is served and metered at zero, because require_pricing is off."}{" "}
      Both switches live on Settings.
    </InfoBanner>
  )
}
