import { Link } from "@tanstack/react-router"

import type { CatalogProviderSummary } from "@/client"
import { EmptyMessage } from "@/design-system/feedback/EmptyMessage"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { ProviderPolicyLinks } from "@/features/providers/ProviderPolicyLinks"
import { publicProvidersHref } from "@/features/providers/publicProviders"
import { useCatalogProviders } from "@/shared/api/models"
import { ProviderMark } from "@/shared/components/marks/BrandMark"
import { formatNumber } from "@/shared/helpers/format"

// Every provider the deployment serves, one card each: its mark and name, how
// many models it offers, and the operator's links to its terms. A card opens the
// provider's own page.
//
// `ProvidersView` takes `publicView` the way `ModelCatalogView` does, so the
// same page renders ahead of a session (`PublicProvidersPage`), where links are
// hash paths because there is no router.

export function modelCountLabel(count: number): string {
  return `${formatNumber(count)} ${count === 1 ? "model" : "models"}`
}

function ProviderCard({
  provider,
  publicView,
}: {
  provider: CatalogProviderSummary
  publicView: boolean
}) {
  // The title link stretches over the card (`after:inset-0`), so the whole
  // card opens the provider while the policy links above it stay their own
  // targets: an anchor cannot hold another.
  const titleClass =
    "text-heading text-link break-words after:absolute after:inset-0 group-hover:text-link-hover"
  return (
    <li className="group relative flex flex-col gap-2 border border-border bg-surface p-4 transition-colors duration-150 ease-out hover:bg-surface-alt motion-reduce:transition-none">
      <div className="flex items-center gap-3">
        <ProviderMark
          providerId={provider.provider}
          label={provider.display_name}
        />
        {publicView ? (
          <a href={publicProvidersHref(provider.name)} className={titleClass}>
            {provider.display_name}
          </a>
        ) : (
          <Link
            to="/providers/$name"
            params={{ name: provider.name }}
            className={titleClass}
          >
            {provider.display_name}
          </Link>
        )}
      </div>
      <span className="text-caption">
        {modelCountLabel(provider.model_count)}
      </span>
      <ProviderPolicyLinks provider={provider} />
    </li>
  )
}

export function ProvidersView({
  publicView = false,
}: {
  publicView?: boolean
}) {
  const providers = useCatalogProviders()
  const rows = providers.data?.providers

  return (
    <div className="flex flex-col gap-5">
      <PageIntro title="Providers">
        {rows
          ? `${formatNumber(rows.length)} ${rows.length === 1 ? "provider serves" : "providers serve"} the models on this deployment.`
          : "The providers that serve the models on this deployment."}
      </PageIntro>
      {providers.isError && !rows ? (
        <ErrorBanner error={providers.error} />
      ) : !rows ? (
        <PageLoading label="Loading providers…" />
      ) : rows.length === 0 ? (
        <EmptyMessage>No provider serves models here yet.</EmptyMessage>
      ) : (
        <ul
          aria-label="Providers"
          className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3"
        >
          {rows.map((provider) => (
            <ProviderCard
              key={provider.name}
              provider={provider}
              publicView={publicView}
            />
          ))}
        </ul>
      )}
    </div>
  )
}

/** The providers list on the router. */
export function ProvidersPage() {
  return <ProvidersView />
}
