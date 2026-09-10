import { Button } from "@heroui/react"
import { Link, useNavigate } from "@tanstack/react-router"
import type { ReactNode } from "react"
import { useState } from "react"
import type { SortDescriptor } from "react-aria-components"
import { FiChevronDown } from "react-icons/fi"

import type { CatalogModelSummary } from "@/client"
import {
  activeFilterCount,
  CAPABILITY_FILTERS,
  type CatalogFilters,
  type CatalogSortColumn,
  CONTEXT_OPTIONS,
  compareModels,
  EMPTY_FILTERS,
  filterModels,
  MODALITIES,
  MODALITY_LABELS,
  PRICE_OPTIONS,
  PRICING_OPTIONS,
  providerOptions,
  RELEASE_OPTIONS,
  SORT_OPTIONS,
  SOURCE_OPTIONS,
  vendorOptions,
} from "@/features/models/catalog"
import { publicCatalogHref } from "@/features/models/publicCatalog"
import { useCatalog } from "@/shared/api/models"
import {
  DataTable,
  type DataTableColumn,
} from "@/shared/components/data/DataTable"
import { TablePagination } from "@/shared/components/data/TablePagination"
import { EmptyMessage } from "@/shared/components/feedback/EmptyMessage"
import { ErrorBanner } from "@/shared/components/feedback/ErrorBanner"
import { PageLoading } from "@/shared/components/feedback/PageLoading"
import { Checkbox } from "@/shared/components/forms/Checkbox"
import { INPUT_CLASS } from "@/shared/components/forms/inputClass"
import { Badge } from "@/shared/components/indicators/Badge"
import { TableScrollFrame } from "@/shared/components/layout/TableScrollFrame"
import { FilterSelect } from "@/shared/components/navigation/FilterSelect"
import { Segmented } from "@/shared/components/navigation/Segmented"
import {
  formatContext,
  formatRate,
  formatRelative,
  formatReleaseDate,
} from "@/shared/helpers/format"
import { useUrlValue } from "@/shared/helpers/urlState"

// The catalog, grouped by model: a rail of filters on the left, and on the
// right a search, a sort, and one card per model. A card is a link to the model's own page, `/models/<id>`,
// where its offerings are compared. Below `lg` the rail folds behind a
// "Filters" button.
//
// Read-only for every caller. A price is set on Model pricing, which the model
// page's links reach with the selector in hand, so the catalog cannot be used
// to reprice anything by accident (otari-ai#2095, #2096).
//
// `ModelCatalogView` is the page with its navigation handed in; `ModelCatalogPage`
// binds it to the router. The split lets the same page render ahead of a
// session as the public catalog (`PublicCatalogPage`), where there is no
// router to link through and no organization to ask about.

const DEFAULT_PAGE_SIZE = 25

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

/** A group in the rail: a heading that folds its choices away. */
function FilterGroup({
  label,
  count,
  defaultOpen = false,
  children,
}: {
  label: string
  /** How many of its choices are in force, shown beside the label. */
  count: number
  defaultOpen?: boolean
  children: ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen || count > 0)
  return (
    <div className="flex flex-col border-b border-border-subtle py-2">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((prev) => !prev)}
        className="flex min-h-9 w-full items-center justify-between gap-2 text-left text-sm text-foreground"
      >
        <span className="flex items-center gap-2">
          {label}
          {count > 0 ? (
            <span className="text-mono-micro text-muted">{count}</span>
          ) : null}
        </span>
        <FiChevronDown
          aria-hidden="true"
          className={`h-4 w-4 shrink-0 text-subtle transition-transform ${open ? "rotate-180" : ""}`}
        />
      </button>
      <div hidden={!open} className="flex flex-col gap-1.5 pb-2">
        {children}
      </div>
    </div>
  )
}

function toggle(list: string[], value: string, on: boolean): string[] {
  return on
    ? [...new Set([...list, value])]
    : list.filter((entry) => entry !== value)
}

/** A one-of-many choice in the rail, as a native radio list. */
function RadioList({
  name,
  value,
  onChange,
  options,
}: {
  name: string
  value: string
  onChange: (value: string) => void
  options: { value: string; label: string }[]
}) {
  return (
    <div role="radiogroup" aria-label={name} className="flex flex-col gap-1">
      {options.map((option) => (
        <label
          key={option.value}
          className="flex min-h-8 cursor-pointer items-center gap-2 text-sm text-foreground"
        >
          <input
            type="radio"
            name={name}
            value={option.value}
            checked={value === option.value}
            onChange={() => onChange(option.value)}
            className="accent-[var(--color-control-indicator)]"
          />
          {option.label}
        </label>
      ))}
    </div>
  )
}

function FilterRail({
  models,
  filters,
  onChange,
}: {
  models: CatalogModelSummary[]
  filters: CatalogFilters
  onChange: (next: CatalogFilters) => void
}) {
  const set = <K extends keyof CatalogFilters>(
    key: K,
    value: CatalogFilters[K],
  ) => onChange({ ...filters, [key]: value })
  return (
    <div className="flex flex-col">
      <FilterGroup
        label="Input modalities"
        count={filters.inputModalities.length}
        defaultOpen
      >
        {MODALITIES.map((modality) => (
          <Checkbox
            key={modality}
            isSelected={filters.inputModalities.includes(modality)}
            onChange={(on) =>
              set(
                "inputModalities",
                toggle(filters.inputModalities, modality, on),
              )
            }
          >
            {MODALITY_LABELS[modality] ?? modality}
          </Checkbox>
        ))}
      </FilterGroup>
      <FilterGroup
        label="Output modalities"
        count={filters.outputModalities.length}
      >
        {MODALITIES.map((modality) => (
          <Checkbox
            key={modality}
            isSelected={filters.outputModalities.includes(modality)}
            onChange={(on) =>
              set(
                "outputModalities",
                toggle(filters.outputModalities, modality, on),
              )
            }
          >
            {MODALITY_LABELS[modality] ?? modality}
          </Checkbox>
        ))}
      </FilterGroup>
      <FilterGroup
        label="Context length"
        count={filters.minContext > 0 ? 1 : 0}
      >
        <RadioList
          name="Context length"
          value={String(filters.minContext)}
          onChange={(value) => set("minContext", Number(value) || 0)}
          options={CONTEXT_OPTIONS}
        />
      </FilterGroup>
      <FilterGroup label="Prompt pricing" count={filters.maxInput > 0 ? 1 : 0}>
        <RadioList
          name="Prompt pricing"
          value={String(filters.maxInput)}
          onChange={(value) => set("maxInput", Number(value) || 0)}
          options={PRICE_OPTIONS}
        />
      </FilterGroup>
      <FilterGroup label="Providers" count={filters.providers.length}>
        {providerOptions(models).map((option) => (
          <Checkbox
            key={option.value}
            isSelected={filters.providers.includes(option.value)}
            onChange={(on) =>
              set("providers", toggle(filters.providers, option.value, on))
            }
          >
            {option.label}
          </Checkbox>
        ))}
      </FilterGroup>
      <FilterGroup label="Vendors" count={filters.vendors.length}>
        {vendorOptions(models).map((option) => (
          <Checkbox
            key={option.value || "unknown"}
            isSelected={filters.vendors.includes(option.value)}
            onChange={(on) =>
              set("vendors", toggle(filters.vendors, option.value, on))
            }
          >
            {option.label}
          </Checkbox>
        ))}
      </FilterGroup>
      <FilterGroup label="Capabilities" count={filters.capabilities.length}>
        {CAPABILITY_FILTERS.map((entry) => (
          <Checkbox
            key={entry.value}
            isSelected={filters.capabilities.includes(entry.value)}
            onChange={(on) =>
              set("capabilities", toggle(filters.capabilities, entry.value, on))
            }
          >
            {entry.label}
          </Checkbox>
        ))}
      </FilterGroup>
      <FilterGroup label="Pricing" count={filters.pricing !== "all" ? 1 : 0}>
        <RadioList
          name="Pricing"
          value={filters.pricing}
          onChange={(value) => set("pricing", value)}
          options={PRICING_OPTIONS}
        />
      </FilterGroup>
      <FilterGroup label="Source" count={filters.source !== "all" ? 1 : 0}>
        <RadioList
          name="Source"
          value={filters.source}
          onChange={(value) => set("source", value)}
          options={SOURCE_OPTIONS}
        />
      </FilterGroup>
      <FilterGroup
        label="Model age"
        count={filters.releasedWithinDays > 0 ? 1 : 0}
      >
        <RadioList
          name="Model age"
          value={String(filters.releasedWithinDays)}
          onChange={(value) => set("releasedWithinDays", Number(value) || 0)}
          options={RELEASE_OPTIONS}
        />
      </FilterGroup>
    </div>
  )
}

/** A separator between the facts on a card's last line. */
function Sep() {
  return (
    <span aria-hidden="true" className="text-subtle">
      |
    </span>
  )
}

function ModelCard({
  model,
  publicView,
  onOpen,
}: {
  model: CatalogModelSummary
  publicView: boolean
  onOpen: (modelId: string) => void
}) {
  const title = model.vendor ? `${model.vendor}: ${model.name}` : model.name
  const titleClass = "text-heading text-link hover:text-link-hover break-words"
  const providers =
    model.provider_count === 1
      ? "1 provider"
      : `${model.provider_count} providers`
  return (
    <article className="flex flex-col gap-2 border border-border bg-surface p-4">
      <div className="flex flex-wrap items-start justify-between gap-x-4 gap-y-1">
        <div className="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1">
          {publicView ? (
            <a href={publicCatalogHref(model.id)} className={titleClass}>
              {title}
            </a>
          ) : (
            <Link
              to="/models/$"
              params={{ _splat: model.id }}
              className={titleClass}
              onClick={(event) => {
                event.preventDefault()
                onOpen(model.id)
              }}
            >
              {title}
            </Link>
          )}
          {model.open_weights ? <Badge tone="muted">Open weights</Badge> : null}
          {model.deprecated ? <Badge tone="warn">Deprecated</Badge> : null}
        </div>
        <span className="shrink-0 text-caption">{providers}</span>
      </div>
      {model.description ? (
        <p className="line-clamp-2 text-sm text-muted">{model.description}</p>
      ) : null}
      <p className="flex flex-wrap items-center gap-x-2 gap-y-1 text-caption">
        <span>by {model.vendor ?? "unknown vendor"}</span>
        {model.release_date ? (
          <>
            <Sep />
            <span>{formatReleaseDate(model.release_date)}</span>
          </>
        ) : null}
        {model.context_window != null ? (
          <>
            <Sep />
            <span>{formatContext(model.context_window)} context</span>
          </>
        ) : null}
        <Sep />
        <span className="tabular-nums">
          {model.min_input_price_per_million == null
            ? "unpriced"
            : `${formatRate(model.min_input_price_per_million)}/M input tokens`}
        </span>
        {model.min_output_price_per_million != null ? (
          <>
            <Sep />
            <span className="tabular-nums">
              {formatRate(model.min_output_price_per_million)}/M output tokens
            </span>
          </>
        ) : null}
      </p>
    </article>
  )
}

function tableColumns(): DataTableColumn<CatalogModelSummary>[] {
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
          </span>
        </div>
      ),
    },
    {
      id: "context",
      header: "Context",
      align: "end",
      allowsSorting: true,
      cell: (row) => (
        <span className="text-mono-caption">
          {formatContext(row.context_window)}
        </span>
      ),
    },
    {
      id: "released",
      header: "Released",
      align: "end",
      allowsSorting: true,
      cell: (row) => (
        <span className="text-mono-caption">
          {formatReleaseDate(row.release_date)}
        </span>
      ),
    },
    {
      id: "input",
      header: "Input / 1M",
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
      header: "Output / 1M",
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

const VIEW_OPTIONS = [
  { value: "list", label: "List" },
  { value: "table", label: "Table" },
]

export function ModelCatalogView({
  onOpen,
  publicView = false,
  initialProvider = "",
}: {
  /** Where a pressed card goes. */
  onOpen: (modelId: string) => void
  /**
   * Ahead of a session: no organization to price for, and plain hash links
   * because there is no router to link through.
   */
  publicView?: boolean
  /** A provider instance to start filtered on. */
  initialProvider?: string
}) {
  const catalog = useCatalog()

  const [filters, setFilters] = useState<CatalogFilters>({
    ...EMPTY_FILTERS,
    providers: initialProvider ? [initialProvider] : [],
  })
  const [sort, setSort] = useState("newest")
  const [view, setView] = useState("list")
  const [railOpen, setRailOpen] = useState(false)
  const [page, setPage] = useState(0)
  const [pageSize, setPageSize] = useState(DEFAULT_PAGE_SIZE)
  const [tableSort, setTableSort] = useState<{
    column: CatalogSortColumn
    direction: "asc" | "desc"
  }>({ column: "name", direction: "asc" })

  const models = catalog.data?.models ?? []
  const sortChoice =
    SORT_OPTIONS.find((option) => option.value === sort) ?? SORT_OPTIONS[0]
  const order =
    view === "table"
      ? compareModels(tableSort.column, tableSort.direction)
      : compareModels(sortChoice.column, sortChoice.direction)
  const filtered = filterModels(models, filters).sort(order)
  const pageCount = Math.max(1, Math.ceil(filtered.length / pageSize))
  const clampedPage = Math.min(page, pageCount - 1)
  const pageRows = filtered.slice(
    clampedPage * pageSize,
    (clampedPage + 1) * pageSize,
  )
  const providerCount = new Set(models.flatMap((model) => model.providers)).size
  const activeCount = activeFilterCount(filters)

  const updateFilters = (next: CatalogFilters) => {
    setFilters(next)
    setPage(0)
  }

  const sortDescriptor: SortDescriptor = {
    column: tableSort.column,
    direction: tableSort.direction === "asc" ? "ascending" : "descending",
  }
  const onSortChange = (descriptor: SortDescriptor) => {
    setTableSort({
      column: String(descriptor.column) as CatalogSortColumn,
      direction: descriptor.direction === "ascending" ? "asc" : "desc",
    })
    setPage(0)
  }

  const defaultsAsOf = catalog.data?.defaults_as_of

  return (
    <div className="flex flex-col gap-5">
      <header className="flex flex-col gap-1">
        <h1 className="text-display">Models</h1>
        <p className="max-w-[38.75rem] text-sm text-muted">
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
        </p>
      </header>

      <ErrorBanner error={catalog.error} />

      <div className="flex flex-col gap-6 lg:grid lg:grid-cols-[14rem_minmax(0,1fr)] lg:gap-8">
        <aside
          aria-label="Filters"
          className={`${railOpen ? "flex" : "hidden"} flex-col lg:flex`}
        >
          <div className="flex items-center justify-between pb-1">
            <span className="text-overline">Filters</span>
            {activeCount > 0 ? (
              <Button
                size="sm"
                variant="ghost"
                onPress={() => updateFilters(EMPTY_FILTERS)}
              >
                Clear
              </Button>
            ) : null}
          </div>
          <FilterRail
            models={models}
            filters={filters}
            onChange={updateFilters}
          />
        </aside>

        <div className="flex min-w-0 flex-col gap-4">
          <div className="otari-toolbar flex flex-wrap items-center gap-2">
            <div className="min-w-[12rem] flex-1">
              <SearchInput
                value={filters.query}
                onChange={(query) => updateFilters({ ...filters, query })}
              />
            </div>
            {view === "list" ? (
              <FilterSelect
                ariaLabel="Sort models"
                value={sort}
                onChange={(value) => {
                  setSort(value)
                  setPage(0)
                }}
                options={SORT_OPTIONS.map(({ value, label }) => ({
                  value,
                  label,
                }))}
              />
            ) : null}
            <Segmented
              label="View"
              options={VIEW_OPTIONS}
              value={view}
              onChange={setView}
            />
            <Button
              size="sm"
              variant="ghost"
              className="lg:hidden"
              aria-expanded={railOpen}
              onPress={() => setRailOpen((prev) => !prev)}
            >
              {railOpen
                ? "Hide filters"
                : activeCount > 0
                  ? `Filters (${activeCount})`
                  : "Filters"}
            </Button>
          </div>

          {catalog.isPending && !catalog.data ? (
            <PageLoading label="Loading models…" />
          ) : view === "table" ? (
            <TableScrollFrame className="otari-models-table">
              <DataTable
                ariaLabel="Models"
                columns={tableColumns()}
                rows={pageRows}
                getRowKey={(row) => row.id}
                sortDescriptor={sortDescriptor}
                onSortChange={onSortChange}
                onRowAction={onOpen}
                emptyContent={
                  <EmptyMessage>
                    {models.length === 0
                      ? "No models yet. Configure a provider, or price a model on Model pricing."
                      : "No models match these filters."}
                  </EmptyMessage>
                }
              />
            </TableScrollFrame>
          ) : pageRows.length === 0 ? (
            <EmptyMessage minHeightClass="min-h-[12rem]">
              {models.length === 0
                ? "No models yet. Configure a provider, or price a model on Model pricing."
                : "No models match these filters."}
            </EmptyMessage>
          ) : (
            <ul aria-label="Models" className="flex flex-col gap-3">
              {pageRows.map((model) => (
                <li key={model.id}>
                  <ModelCard
                    model={model}
                    publicView={publicView}
                    onOpen={onOpen}
                  />
                </li>
              ))}
            </ul>
          )}
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
      </div>
    </div>
  )
}

/** The catalog on the router: a pressed card navigates to the model's page. */
export function ModelCatalogPage() {
  const navigate = useNavigate()
  // A provider clicked on the Providers page arrives as ?provider=<instance>,
  // pre-selecting that provider's filter so the list shows only its models.
  const providerParam = useUrlValue("provider")
  return (
    <ModelCatalogView
      initialProvider={providerParam}
      onOpen={(id) => {
        void navigate({ to: "/models/$", params: { _splat: id } })
      }}
    />
  )
}
