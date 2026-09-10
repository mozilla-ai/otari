import { Link, useNavigate } from "@tanstack/react-router"
import { useState } from "react"
import type { SortDescriptor } from "react-aria-components"

import type { CatalogModelSummary } from "@/client"
import {
  CAPABILITY_FILTERS,
  type CatalogSortColumn,
  COMPARE_AT_OPTIONS,
  CONTEXT_OPTIONS,
  compareModels,
  filterModels,
  PRICE_OPTIONS,
  PRICING_OPTIONS,
  providerOptions,
  RELEASE_OPTIONS,
  SOURCE_OPTIONS,
  vendorOptions,
} from "@/features/models/catalog"
import { ModelDetailPanel } from "@/features/models/ModelDetailPanel"
import { publicCatalogHref } from "@/features/models/publicCatalog"
import { canManage, isDeploymentOperator } from "@/features/organization/roles"
import { useCatalog, useCatalogModel } from "@/shared/api/models"
import { useOrganizationContext } from "@/shared/api/organizations"
import {
  DataTable,
  type DataTableColumn,
} from "@/shared/components/data/DataTable"
import { TablePagination } from "@/shared/components/data/TablePagination"
import { EmptyMessage } from "@/shared/components/feedback/EmptyMessage"
import { ErrorBanner } from "@/shared/components/feedback/ErrorBanner"
import { INPUT_CLASS } from "@/shared/components/forms/inputClass"
import { PageIntro } from "@/shared/components/layout/PageIntro"
import { TableScrollFrame } from "@/shared/components/layout/TableScrollFrame"
import { Toolbar } from "@/shared/components/layout/Toolbar"
import { FilterSelect } from "@/shared/components/navigation/FilterSelect"
import {
  formatContext,
  formatRate,
  formatRelative,
} from "@/shared/helpers/format"
import { useUrlValue } from "@/shared/helpers/urlState"

// The catalog, grouped by model.
//
// Two columns at 1:1.75, each static on the page and scrolling on its own
// (otari-ai#2109, #2112): the list of models on the left, the selected model's
// offerings on the right. The selection is the route, `/models/$modelId`, so a
// model is a place somebody can be sent to; below `lg` the columns stack and
// the one that has something to say is shown.
//
// Read-only for every caller. A price is set on Model pricing, which the
// detail's "Edit rate" link reaches with the selector in hand, so the catalog
// cannot be used to reprice anything by accident (otari-ai#2095, #2096).
//
// `ModelCatalogView` is the page with its navigation handed in; `ModelCatalogPage`
// binds it to the router. The split lets the same page render ahead of a
// session as the public catalog (`PublicCatalogPage`), where there is no
// router to link through and no organization to ask about.

const DEFAULT_PAGE_SIZE = 25
const CAPABILITY_OPTIONS = [
  { value: "all", label: "Any capability" },
  ...CAPABILITY_FILTERS.map((entry) => ({
    value: entry.value,
    label: entry.label,
  })),
]

function SearchInput({
  value,
  onChange,
}: {
  value: string
  onChange: (value: string) => void
}) {
  return (
    <input
      type="search"
      value={value}
      onChange={(event) => onChange(event.target.value)}
      placeholder="Search models…"
      aria-label="Search models"
      className={`w-full min-w-0 ${INPUT_CLASS}`}
    />
  )
}

function fromRate(value: number | null | undefined): string {
  return value == null ? "—" : `from ${formatRate(value)}`
}

function columns(atContext: number): DataTableColumn<CatalogModelSummary>[] {
  // The rate lanes say what size they compare at, since the same offering is a
  // different price at 8K and at 500K once it is tiered.
  const at = atContext ? ` at ${formatContext(atContext)}` : ""
  return [
    {
      id: "name",
      header: "Model",
      isRowHeader: true,
      allowsSorting: true,
      cell: (row) => (
        <div className="flex min-w-0 flex-col">
          <span className="text-body break-words">{row.name}</span>
          <span className="text-caption">
            {row.vendor ?? "Unknown vendor"} ·{" "}
            {row.provider_count === 1
              ? "1 provider"
              : `${row.provider_count} providers`}
            {row.context_window != null
              ? ` · up to ${formatContext(row.context_window)}`
              : ""}
          </span>
        </div>
      ),
    },
    {
      id: "input",
      header: `Input / 1M${at}`,
      align: "end",
      allowsSorting: true,
      cell: (row) => (
        <span className="text-mono-caption">
          {fromRate(row.min_input_price_per_million)}
        </span>
      ),
    },
    {
      id: "output",
      header: `Output / 1M${at}`,
      align: "end",
      allowsSorting: true,
      cell: (row) => (
        <span className="text-mono-caption">
          {fromRate(row.min_output_price_per_million)}
        </span>
      ),
    },
  ]
}

export function ModelCatalogView({
  modelId,
  onOpen,
  publicView = false,
  initialProvider = "",
}: {
  modelId?: string
  /** Where a pressed row goes. */
  onOpen: (modelId: string) => void
  /**
   * Ahead of a session: no organization to price for, no operator affordance,
   * and plain hash links because there is no router to link through.
   */
  publicView?: boolean
  /** A provider instance to start filtered on. */
  initialProvider?: string
}) {
  const organization = useOrganizationContext(!publicView)
  const isOperator = !publicView && isDeploymentOperator(organization.data)
  const canOverride = !publicView && canManage(organization.data)
  const [compareAt, setCompareAt] = useState("0")
  const atContext = Number(compareAt) || 0
  const catalog = useCatalog(atContext || null)
  const selected = useCatalogModel(modelId)

  const [search, setSearch] = useState("")
  const [vendor, setVendor] = useState("all")
  const [provider, setProvider] = useState(initialProvider || "all")
  const [capability, setCapability] = useState("all")
  const [minContext, setMinContext] = useState("0")
  const [pricing, setPricing] = useState("all")
  const [source, setSource] = useState("all")
  const [maxInput, setMaxInput] = useState("0")
  const [release, setRelease] = useState("0")
  const [page, setPage] = useState(0)
  const [pageSize, setPageSize] = useState(DEFAULT_PAGE_SIZE)
  const [sort, setSort] = useState<{
    column: CatalogSortColumn
    direction: "asc" | "desc"
  }>({ column: "name", direction: "asc" })

  const models = catalog.data?.models ?? []
  const filtered = filterModels(models, {
    query: search,
    vendor,
    provider,
    capability,
    minContext: Number(minContext) || 0,
    pricing,
    source,
    maxInput: Number(maxInput) || 0,
    releasedWithinDays: Number(release) || 0,
  }).sort(compareModels(sort.column, sort.direction))
  const pageCount = Math.max(1, Math.ceil(filtered.length / pageSize))
  const clampedPage = Math.min(page, pageCount - 1)
  const pageRows = filtered.slice(
    clampedPage * pageSize,
    (clampedPage + 1) * pageSize,
  )
  const providerCount = new Set(models.flatMap((model) => model.providers)).size

  const resetPage =
    <T,>(setter: (value: T) => void) =>
    (value: T) => {
      setter(value)
      setPage(0)
    }

  const sortDescriptor: SortDescriptor = {
    column: sort.column,
    direction: sort.direction === "asc" ? "ascending" : "descending",
  }
  const onSortChange = (descriptor: SortDescriptor) => {
    setSort({
      column: String(descriptor.column) as CatalogSortColumn,
      direction: descriptor.direction === "ascending" ? "asc" : "desc",
    })
    setPage(0)
  }

  const defaultsAsOf = catalog.data?.defaults_as_of

  return (
    <>
      <PageIntro title="Models">
        {catalog.data ? (
          <>
            {models.length} {models.length === 1 ? "model" : "models"} across{" "}
            {providerCount} {providerCount === 1 ? "provider" : "providers"}.{" "}
            {publicView
              ? "Prices are this deployment's list rates, cheapest offering first."
              : "Prices are what your organization is charged, cheapest offering first."}{" "}
            {defaultsAsOf
              ? `Default rates as of ${formatRelative(defaultsAsOf)}.`
              : catalog.data.default_pricing
                ? "Default rates come from the bundled genai-prices dataset."
                : "Default pricing is off: a model with no stored rate is unpriced."}
          </>
        ) : (
          "Every model this deployment can serve, grouped by model, with each provider's offering and price."
        )}
      </PageIntro>

      <ErrorBanner error={catalog.error} />

      {/* Two static columns, each its own scroll region, at 1:1.75. Sticky
          rather than a fixed-height page, because the shell's <main> is the
          document's scroll container and a band has no height to fill; pinned
          to the top with a viewport-bounded height, each column scrolls
          internally while the page under them stays put. */}
      <div className="grid gap-6 lg:grid-cols-[1fr_1.75fr] lg:items-start">
        <div
          className={`flex min-w-0 flex-col gap-3 lg:sticky lg:top-0 lg:max-h-[calc(100dvh-8rem)] lg:overflow-y-auto ${
            modelId ? "hidden lg:flex" : ""
          }`}
        >
          <Toolbar>
            <SearchInput value={search} onChange={resetPage(setSearch)} />
            <FilterSelect
              ariaLabel="Filter by provider"
              value={provider}
              onChange={resetPage(setProvider)}
              options={providerOptions(models)}
            />
            <FilterSelect
              ariaLabel="Filter by vendor"
              value={vendor}
              onChange={resetPage(setVendor)}
              options={vendorOptions(models)}
            />
            <FilterSelect
              ariaLabel="Filter by capability"
              value={capability}
              onChange={resetPage(setCapability)}
              options={CAPABILITY_OPTIONS}
            />
            <FilterSelect
              ariaLabel="Minimum context"
              value={minContext}
              onChange={resetPage(setMinContext)}
              options={CONTEXT_OPTIONS}
            />
            <FilterSelect
              ariaLabel="Filter by pricing"
              value={pricing}
              onChange={resetPage(setPricing)}
              options={PRICING_OPTIONS}
            />
            <FilterSelect
              ariaLabel="Filter by source"
              value={source}
              onChange={resetPage(setSource)}
              options={SOURCE_OPTIONS}
            />
            <FilterSelect
              ariaLabel="Maximum input price"
              value={maxInput}
              onChange={resetPage(setMaxInput)}
              options={PRICE_OPTIONS}
            />
            <FilterSelect
              ariaLabel="Release date"
              value={release}
              onChange={resetPage(setRelease)}
              options={RELEASE_OPTIONS}
            />
            <FilterSelect
              ariaLabel="Compare prices at"
              value={compareAt}
              onChange={resetPage(setCompareAt)}
              options={COMPARE_AT_OPTIONS}
            />
          </Toolbar>
          <TableScrollFrame className="otari-models-table">
            <DataTable
              ariaLabel="Models"
              columns={columns(atContext)}
              rows={pageRows}
              getRowKey={(row) => row.id}
              isLoading={catalog.isPending && !catalog.data}
              sortDescriptor={sortDescriptor}
              onSortChange={onSortChange}
              onRowAction={onOpen}
              rowClassName={(row) =>
                row.id === modelId ? "bg-primary-subtle" : undefined
              }
              emptyContent={
                <EmptyMessage>
                  {models.length === 0
                    ? "No models yet. Configure a provider, or price a model on Model pricing."
                    : "No models match these filters."}
                </EmptyMessage>
              }
            />
          </TableScrollFrame>
          <TablePagination
            page={clampedPage}
            pageSize={pageSize}
            total={filtered.length}
            rowsOnPage={pageRows.length}
            onPageChange={setPage}
            onPageSizeChange={(size) => {
              setPageSize(size)
              setPage(0)
            }}
          />
        </div>

        <aside
          aria-label="Model details"
          className={`min-w-0 lg:sticky lg:top-0 lg:max-h-[calc(100dvh-8rem)] lg:overflow-y-auto ${
            modelId ? "" : "hidden lg:block"
          }`}
        >
          {modelId ? (
            <div className="flex flex-col gap-4">
              {publicView ? (
                <a
                  href={publicCatalogHref()}
                  className="text-caption text-link hover:text-link-hover lg:hidden"
                >
                  ← All models
                </a>
              ) : (
                <Link
                  to="/models"
                  className="text-caption text-link hover:text-link-hover lg:hidden"
                >
                  ← All models
                </Link>
              )}
              <ModelDetailPanel
                model={selected.data}
                isLoading={selected.isPending}
                error={selected.error}
                canPrice={isOperator}
                canOverride={canOverride}
                publicView={publicView}
                defaultPricing={catalog.data?.default_pricing}
              />
            </div>
          ) : (
            <EmptyMessage minHeightClass="min-h-[16rem]">
              Select a model to compare the providers offering it.
            </EmptyMessage>
          )}
        </aside>
      </div>
    </>
  )
}

/** The catalog on the router: a pressed row navigates to `/models/$modelId`. */
export function ModelCatalogPage({ modelId }: { modelId?: string }) {
  const navigate = useNavigate()
  // A provider clicked on the Providers page arrives as ?provider=<instance>,
  // pre-selecting that provider's filter so the list shows only its models.
  const providerParam = useUrlValue("provider")
  return (
    <ModelCatalogView
      modelId={modelId}
      initialProvider={providerParam}
      onOpen={(id) => {
        void navigate({ to: "/models/$modelId", params: { modelId: id } })
      }}
    />
  )
}
