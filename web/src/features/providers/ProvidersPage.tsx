import { Link } from "@tanstack/react-router"

import type { CatalogProviderSummary } from "@/client"
import { TablePagination } from "@/design-system/data/TablePagination"
import { EmptyMessage } from "@/design-system/feedback/EmptyMessage"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { ProviderPolicyLinks } from "@/features/providers/ProviderPolicyLinks"
import {
  catalogPage,
  needsPager,
  PROVIDERS_URL_DEFAULTS,
  type ProvidersListState,
  providersListFromUrl,
  providersListToUrl,
  useClampedPage,
} from "@/features/providers/providersList"
import { publicProvidersHref } from "@/features/providers/publicProviders"
import { useCatalogProviders } from "@/shared/api/models"
import { ProviderMark } from "@/shared/components/marks/BrandMark"
import { followInAppLink } from "@/shared/helpers/appLocation"
import { formatNumber } from "@/shared/helpers/format"
import { useUrlState } from "@/shared/helpers/urlState"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"

// Every provider the deployment serves, one card each: its mark and name, how
// many models it offers, and the operator's links to its terms. A card opens the
// provider's own page.
//
// `ProvidersView` takes `publicView` and its page state the way
// `ModelCatalogView` does, so the same page renders ahead of a session
// (`PublicProvidersPage`), where links are plain anchors because there is no
// router and the page lives in component state.

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
    "min-w-0 text-heading text-link wrap-anywhere after:absolute after:inset-0 group-hover:text-link-hover"
  return (
    <li className="group relative flex flex-col gap-2 border border-border bg-surface p-4 transition-colors duration-150 ease-out hover:bg-surface-alt motion-reduce:transition-none">
      <div className="flex items-center gap-3">
        <ProviderMark
          providerId={provider.provider}
          label={provider.display_name}
        />
        {publicView ? (
          <a
            href={publicProvidersHref(provider.name)}
            onClick={followInAppLink}
            className={titleClass}
          >
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
  list,
  onListChange,
  publicView = false,
}: {
  list: ProvidersListState
  onListChange: (next: ProvidersListState) => void
  publicView?: boolean
}) {
  // What the selected workspace may call, held until the selection is known so
  // the page never shows the wider list first. A visitor has no workspace.
  const workspace = useSelectedWorkspace()
  const providers = useCatalogProviders(
    catalogPage(list),
    publicView ? "" : (workspace.selected?.workspace_id ?? ""),
    publicView || workspace.isSettled,
  )
  const rows = providers.data?.providers
  const count = providers.data?.count
  useClampedPage(list, onListChange, count, providers.isPlaceholderData)

  return (
    <div className="flex flex-col gap-5">
      <PageIntro title="Providers">
        {count !== undefined
          ? `${formatNumber(count)} ${count === 1 ? "provider serves" : "providers serve"} the models on this deployment.`
          : "The providers that serve the models on this deployment."}
      </PageIntro>
      {providers.isError ? <ErrorBanner error={providers.error} /> : null}
      {!rows ? (
        providers.isError ? null : (
          <PageLoading label="Loading providers…" />
        )
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
      {count !== undefined && needsPager(count) ? (
        <TablePagination
          page={list.page}
          pageSize={list.pageSize}
          total={count}
          rowsOnPage={rows?.length ?? 0}
          isFetching={providers.isFetching}
          onPageChange={(page) => onListChange({ ...list, page })}
          onPageSizeChange={(pageSize) => onListChange({ page: 0, pageSize })}
        />
      ) : null}
    </div>
  )
}

/** The providers list on the router, its page kept in the URL. */
export function ProvidersPage() {
  const url = useUrlState(PROVIDERS_URL_DEFAULTS)
  return (
    <ProvidersView
      list={providersListFromUrl(url)}
      onListChange={(next) => url.patch(providersListToUrl(next))}
    />
  )
}
