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
import {
  catalogPage,
  needsPager,
  type ProvidersListState,
  useClampedPage,
} from "@/features/providers/providersList"
import { publicProvidersHref } from "@/features/providers/publicProviders"
import { ApiError } from "@/shared/api/client"
import { useCatalogProvider } from "@/shared/api/models"
import { ProviderMark } from "@/shared/components/marks/BrandMark"
import { followInAppLink } from "@/shared/helpers/appLocation"
import { formatContext } from "@/shared/helpers/format"

// One provider on a page of its own: its mark, name and the operator's links
// to its terms, then the models it offers with their list prices, a page at a
// time. A model's name opens the model's page, where its offerings across
// providers are compared. Rendered ahead of a session (`PublicProvidersPage`),
// so links are plain anchors.

function price(
  value: number | null | undefined,
  unit: CatalogProviderModel["price_unit"],
): string {
  // An em dash rather than $0.00: a missing rate is not a free one.
  return value == null ? "—" : formatPricedRate(value, unit ?? "token")
}

function modelColumns(): DataTableColumn<CatalogProviderModel>[] {
  return [
    {
      id: "name",
      header: "Model",
      isRowHeader: true,
      cell: (row) => (
        <span className="flex min-w-0 flex-col">
          <a
            href={publicCatalogHref(row.id)}
            onClick={followInAppLink}
            className="text-body break-words text-link hover:text-link-hover"
          >
            {row.name}
          </a>
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

function BackLink() {
  return (
    <nav aria-label="Breadcrumb" className="text-caption">
      <a
        href={publicProvidersHref()}
        onClick={followInAppLink}
        className="text-link hover:text-link-hover"
      >
        ← Providers
      </a>
    </nav>
  )
}

export function ProviderDetailView({
  name,
  list,
  onListChange,
}: {
  name: string
  /** The page of the provider's models on screen. */
  list: ProvidersListState
  onListChange: (next: ProvidersListState) => void
}) {
  // A visitor has no workspace, so the models are the whole catalog's.
  const selected = useCatalogProvider(name, catalogPage(list), "", true)
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
          <BackLink />
          <PageIntro
            title="Provider not found"
            detail={
              <>No provider named {name} serves models on this deployment.</>
            }
          />
        </div>
      )
    }
    if (selected.isError) return <ErrorBanner error={selected.error} />
    return <PageLoading label="Loading provider…" />
  }

  return (
    <div className="flex flex-col gap-6">
      <BackLink />
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
        />
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
            columns={modelColumns()}
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
