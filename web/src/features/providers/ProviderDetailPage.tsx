import { Link } from "@tanstack/react-router"

import type { CatalogProviderModel } from "@/client"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { TablePagination } from "@/design-system/data/TablePagination"
import { EmptyMessage } from "@/design-system/feedback/EmptyMessage"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import { formatPricedRate } from "@/features/models/pricingUnit"
import { publicCatalogHref } from "@/features/models/publicCatalog"
import { ProviderPolicyLinks } from "@/features/providers/ProviderPolicyLinks"
import { modelCountLabel } from "@/features/providers/ProvidersPage"
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
import { ApiError } from "@/shared/api/client"
import { useCatalogProvider } from "@/shared/api/models"
import { ProviderMark } from "@/shared/components/marks/BrandMark"
import { formatContext } from "@/shared/helpers/format"
import { useUrlState } from "@/shared/helpers/urlState"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"

// One provider on a page of its own: its mark, name and the operator's links
// to its terms, then the models it offers with their list prices, a page at a
// time. A model's name opens the model's page, where its offerings across
// providers are compared.

function price(
  value: number | null | undefined,
  unit: CatalogProviderModel["price_unit"],
): string {
  // An em dash rather than $0.00: a missing rate is not a free one.
  return value == null ? "—" : formatPricedRate(value, unit ?? "token")
}

function modelColumns(
  publicView: boolean,
): DataTableColumn<CatalogProviderModel>[] {
  return [
    {
      id: "name",
      header: "Model",
      isRowHeader: true,
      cell: (row) => (
        <span className="flex min-w-0 flex-col">
          {publicView ? (
            <a
              href={publicCatalogHref(row.id)}
              className="text-body break-words text-link hover:text-link-hover"
            >
              {row.name}
            </a>
          ) : (
            <Link
              to="/models/$"
              params={{ _splat: row.id }}
              className="text-body break-words text-link hover:text-link-hover"
            >
              {row.name}
            </Link>
          )}
          <code className="text-caption font-mono max-md:[overflow-wrap:anywhere]">
            {row.id}
          </code>
        </span>
      ),
    },
    {
      id: "vendor",
      header: "Vendor",
      cell: (row) => <span className="text-body">{row.vendor ?? "—"}</span>,
    },
    {
      id: "input",
      header: "Input",
      align: "end",
      cell: (row) => (
        <span className="text-mono-caption whitespace-nowrap">
          {price(row.input_price, row.price_unit)}
        </span>
      ),
    },
    {
      id: "output",
      header: "Output",
      align: "end",
      cell: (row) => (
        <span className="text-mono-caption whitespace-nowrap">
          {price(row.output_price, row.price_unit)}
        </span>
      ),
    },
    {
      id: "context",
      header: "Context",
      align: "end",
      cell: (row) => (
        <span className="text-mono-caption">
          {formatContext(row.context_length)}
        </span>
      ),
    },
  ]
}

const rowKey = (row: CatalogProviderModel) => row.id

function BackLink({ publicView }: { publicView: boolean }) {
  const className = "text-link hover:text-link-hover"
  return (
    <nav aria-label="Breadcrumb" className="text-caption">
      {publicView ? (
        <a href={publicProvidersHref()} className={className}>
          ← Providers
        </a>
      ) : (
        <Link to="/providers" className={className}>
          ← Providers
        </Link>
      )}
    </nav>
  )
}

export function ProviderDetailView({
  name,
  list,
  onListChange,
  publicView = false,
}: {
  name: string
  /** The page of the provider's models on screen. */
  list: ProvidersListState
  onListChange: (next: ProvidersListState) => void
  /** Ahead of a session: links are hash paths, as there is no router. */
  publicView?: boolean
}) {
  // The models the selected workspace may call, as the Models page shows them.
  const workspace = useSelectedWorkspace()
  const selected = useCatalogProvider(
    name,
    catalogPage(list),
    publicView ? "" : (workspace.selected?.workspace_id ?? ""),
    publicView || workspace.isSettled,
  )
  const provider = selected.data
  useClampedPage(
    list,
    onListChange,
    provider?.model_count,
    selected.isPlaceholderData,
  )

  if (!provider) {
    if (
      selected.isError &&
      selected.error instanceof ApiError &&
      selected.error.status === 404
    ) {
      return (
        <div className="flex flex-col gap-6">
          <BackLink publicView={publicView} />
          <PageIntro title="Provider not found">
            No provider named {name} serves models on this deployment.
          </PageIntro>
        </div>
      )
    }
    if (selected.isError) return <ErrorBanner error={selected.error} />
    return <PageLoading label="Loading provider…" />
  }

  return (
    <div className="flex flex-col gap-6">
      <BackLink publicView={publicView} />
      {selected.isError ? <ErrorBanner error={selected.error} /> : null}
      <div className="flex flex-col gap-2">
        <PageIntro
          title={provider.display_name}
          leading={
            <ProviderMark
              providerId={provider.provider}
              label={provider.display_name}
            />
          }
        >
          {modelCountLabel(provider.model_count)}. Token prices are per million
          tokens.
        </PageIntro>
        <ProviderPolicyLinks provider={provider} />
      </div>
      <section
        aria-labelledby="provider-models-title"
        className="flex flex-col gap-3"
      >
        <h2 id="provider-models-title" className="text-heading">
          Models
        </h2>
        <TableScrollFrame className="otari-provider-models-table">
          <DataTable
            ariaLabel={`Models from ${provider.display_name}`}
            columns={modelColumns(publicView)}
            rows={provider.models}
            getRowKey={rowKey}
            mobileLayout="cards"
            emptyContent={
              <EmptyMessage>This provider offers no models yet.</EmptyMessage>
            }
          />
        </TableScrollFrame>
        {needsPager(provider.model_count) ? (
          <TablePagination
            page={list.page}
            pageSize={list.pageSize}
            total={provider.model_count}
            rowsOnPage={provider.models.length}
            isFetching={selected.isFetching}
            onPageChange={(page) => onListChange({ ...list, page })}
            onPageSizeChange={(pageSize) => onListChange({ page: 0, pageSize })}
          />
        ) : null}
      </section>
    </div>
  )
}

/** The provider page on the router, its page of models kept in the URL. */
export function ProviderDetailPage({ name }: { name: string }) {
  const url = useUrlState(PROVIDERS_URL_DEFAULTS)
  return (
    <ProviderDetailView
      name={name}
      list={providersListFromUrl(url)}
      onListChange={(next) => url.patch(providersListToUrl(next))}
    />
  )
}
