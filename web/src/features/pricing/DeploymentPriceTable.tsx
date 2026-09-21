/**
 * The deployment's stored rates: what this gateway meters a request at.
 *
 * Read by any signed-in caller (`/pricing` is `verify_catalog_reader`), and
 * written only by a deployment operator, which is what `canPrice` carries. A
 * non-operator gets the same table without the drift column, the row editor or
 * the ad-hoc pricing dialog, rather than a page of refusals.
 *
 * `?model=<selector>` opens the editor on one row. That is a contract the Models
 * detail page links into, so the parameter name is not this component's to
 * change.
 */

import { Button } from "@heroui/react"
import { Link, useNavigate } from "@tanstack/react-router"
import { useState } from "react"

import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { TablePagination } from "@/design-system/data/TablePagination"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { Section } from "@/design-system/layout/Section"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import {
  type ManualRates,
  SetPriceDialog,
} from "@/features/models/SetPriceDialog"
import { PriceEditor } from "@/features/pricing/PriceEditor"
import {
  currentRows,
  DRIFT_NOTICE_PERCENT,
  formatDrift,
  type PriceRow,
} from "@/features/pricing/priceRows"
import { UNIT_LABELS } from "@/features/pricing/units"
import {
  useCurrentPricing,
  useModelPricing,
  usePricingDrift,
  useSetPricing,
} from "@/shared/api/pricing"
import { formatRate, formatRelative } from "@/shared/helpers/format"
import { useUrlValue } from "@/shared/helpers/urlState"

const DEFAULT_PAGE_SIZE = 25

function DriftCell({ row }: { row: PriceRow }) {
  const drift = row.drift
  if (!drift || drift.default_input_price_per_million == null) {
    return <span className="text-subtle">—</span>
  }
  const worst = Math.max(
    Math.abs(drift.input_delta_percent ?? 0),
    Math.abs(drift.output_delta_percent ?? 0),
  )
  const ink = worst > DRIFT_NOTICE_PERCENT ? "text-danger" : "text-muted"
  return (
    <span
      className={`${ink} tabular-nums`}
      title={`Default today: ${formatRate(drift.default_input_price_per_million)} in, ${formatRate(
        drift.default_output_price_per_million ?? 0,
      )} out${drift.default_reference ? `, from ${drift.default_reference}` : ""}`}
    >
      {formatDrift(drift.input_delta_percent)} /{" "}
      {formatDrift(drift.output_delta_percent)}
    </span>
  )
}

const COLUMNS: DataTableColumn<PriceRow>[] = [
  {
    id: "modelKey",
    header: "Model",
    isRowHeader: true,
    cell: (row) => <span className="font-mono text-body">{row.modelKey}</span>,
  },
  {
    id: "input",
    header: "Input / 1M",
    align: "end",
    cell: (row) => formatRate(row.input),
  },
  {
    id: "output",
    header: "Output / 1M",
    align: "end",
    cell: (row) => formatRate(row.output),
  },
  {
    id: "cacheRead",
    header: "Cache read / 1M",
    align: "end",
    // An em dash rather than $0.00: a model with no cache-read rate is not the
    // same as one that reads cache for free.
    cell: (row) => (row.cacheRead === null ? "—" : formatRate(row.cacheRead)),
  },
  {
    id: "tiers",
    header: "Tiers",
    align: "end",
    cell: (row) => (row.tiers === 0 ? "—" : `${row.tiers} configured`),
  },
  {
    id: "unit",
    header: "Per 1M",
    cell: (row) => (
      <span className="text-muted">{UNIT_LABELS[row.unit] ?? row.unit}</span>
    ),
  },
  // Where the rate was set: the config file or the API. A rate set in config
  // comes back on every restart, and knowing that before editing it here is
  // the difference between a change that sticks and one that does not.
  {
    id: "origin",
    header: "Set by",
    cell: (row) => (
      <span className="text-muted">
        {row.origin === "config"
          ? "config"
          : row.origin === "api"
            ? "dashboard"
            : "—"}
      </span>
    ),
  },
  // The row has carried this since it was built and never rendered it. It earns
  // the lane now because something has to absorb the width this table does not
  // use, and the alternative was a gap: when a rate last moved is the question
  // an operator brings to a price they did not expect.
  {
    id: "updatedAt",
    header: "Updated",
    align: "end",
    cell: (row) => (
      <span className="text-muted">{formatRelative(row.updatedAt)}</span>
    ),
  },
  // Unlabelled and empty, and it is the only lane in either table that is.
  // Something has to absorb the width a table does not use, and every candidate
  // that carries data is the wrong one: a date given the slack rendered in a
  // 985px lane, a value floating alone in a field with a hit area to match, and
  // handing it to the model key instead threw the rates against the right edge.
  // A lane with nothing in it can take any width without lying about anything,
  // which also keeps the rate lanes at the widths the overrides table below
  // uses, so the two still line up.
  { id: "spacer", header: "", cell: () => null },
]

/**
 * The catalog rows, and where a rate is edited when the caller may edit one.
 *
 * `canPrice` is the caller's, not the deployment's: setting a catalog rate is
 * `POST /v1/pricing`, which is operator-only, and since otari#867 a non-operator
 * reads Models with every pricing affordance gone. So both sentences that point
 * at that editor are the operator's, and for anyone else the empty table says
 * only what is true, that nothing is priced. Passed in rather than resolved here
 * so the page asks who is calling once and no two sections can answer it
 * differently.
 */
export function DeploymentPriceTable({ canPrice }: { canPrice: boolean }) {
  const [page, setPage] = useState(0)
  const [pageSize, setPageSize] = useState(DEFAULT_PAGE_SIZE)
  const pricing = useCurrentPricing(page, pageSize)
  // Operator-only read, so it is gated on the same axis as the editor.
  const drift = usePricingDrift(canPrice)
  const navigate = useNavigate()
  const setPricing = useSetPricing()
  // The selector whose deployment rate is being edited, carried in the URL so
  // the catalog's "Edit rate" link lands here with it in hand.
  const editingKey = useUrlValue("model")
  const [customOpen, setCustomOpen] = useState(false)
  // Bumped on every open and used as the dialog's key: it seeds its draft on
  // mount and owns the refusal, so a remount is what clears both.
  const [customOpenCount, setCustomOpenCount] = useState(0)
  const live = pricing.data?.data ?? []
  const rows = currentRows(live, drift.data ?? [])
  // The editor opens on a key linked from Models, which need not be on the page
  // being shown, so the row is read from the page where it is there and fetched
  // by key where it is not.
  const onPage = live.find((row) => row.model_key === editingKey)
  const fetched = useModelPricing(editingKey && !onPage ? editingKey : null)
  const current = onPage ?? fetched.data ?? undefined

  const edit = (modelKey: string | null) =>
    void navigate({
      to: "/organization/pricing",
      search: modelKey ? { model: modelKey } : {},
    })

  // A backend with no /v1/models endpoint serves models the catalog never
  // lists, so the only way to meter them is a key typed by hand. The stored key
  // is what the server normalized, so the editor opens on that rather than on
  // the raw input.
  const priceCustom = async (rates: ManualRates, modelKey: string) => {
    const created = await setPricing.mutateAsync({
      model_key: modelKey,
      input_price_per_million: rates.input_price_per_million,
      output_price_per_million: rates.output_price_per_million,
      cache_read_price_per_million: rates.cache_read_price_per_million ?? null,
      cache_write_price_per_million:
        rates.cache_write_price_per_million ?? null,
    })
    setCustomOpen(false)
    edit(created.model_key)
  }

  const columns = canPrice
    ? [
        ...COLUMNS.filter((column) => column.id !== "spacer"),
        {
          id: "drift",
          header: "vs default",
          align: "end" as const,
          cell: (row: PriceRow) => <DriftCell row={row} />,
        },
        {
          id: "actions",
          header: "Actions",
          cell: (row: PriceRow) => (
            <Link
              to="/organization/pricing"
              search={{ model: row.modelKey }}
              className="text-link hover:text-link-hover"
            >
              Edit
            </Link>
          ),
        },
      ]
    : COLUMNS

  if (pricing.isLoading) return <PageLoading label="Loading model prices…" />

  return (
    <>
      {/* The group's heading and the rule under it are what introduce the rows,
          which then sit straight on the page ground. No box: the header rule
          and the row separators already say where the group starts and ends. */}
      <Section
        className="pt-6 pb-3"
        contentClassName="flex flex-wrap items-center justify-between gap-3"
      >
        <h2 className="text-title">Model prices</h2>
        {canPrice ? (
          <Button
            size="sm"
            variant="ghost"
            onPress={() => {
              setCustomOpenCount((count) => count + 1)
              setCustomOpen(true)
            }}
          >
            Price a model
          </Button>
        ) : null}
      </Section>
      <ErrorBanner error={pricing.error} />
      {canPrice && editingKey ? (
        <Section
          aria-labelledby="price-editor-title"
          className="border-y border-border py-5"
          contentClassName="flex flex-col gap-3"
        >
          <div className="flex items-center justify-between gap-3">
            <h3 id="price-editor-title" className="text-title break-all">
              {current ? "Edit price for " : "Set price for "}
              <code className="text-mono-title">{editingKey}</code>
            </h3>
            <Button size="sm" variant="ghost" onPress={() => edit(null)}>
              Close
            </Button>
          </div>
          <div className="max-w-xl">
            <PriceEditor
              key={editingKey}
              modelKey={editingKey}
              current={current}
              onDone={() => edit(null)}
            />
          </div>
        </Section>
      ) : null}
      <TableScrollFrame className="otari-pricing-table">
        <DataTable
          ariaLabel="Model prices"
          columns={columns}
          rows={rows}
          getRowKey={(row) => row.modelKey}
          emptyContent={
            canPrice
              ? "No model carries a stored price yet. Pick one on Models, or price one by its selector here."
              : "No model carries a stored price yet."
          }
        />
      </TableScrollFrame>
      <TablePagination
        page={page}
        pageSize={pageSize}
        total={pricing.data?.count ?? null}
        rowsOnPage={rows.length}
        onPageChange={setPage}
        onPageSizeChange={(size) => {
          setPageSize(size)
          setPage(0)
        }}
        isFetching={pricing.isFetching}
        label="model prices"
      />
      {canPrice ? (
        <SetPriceDialog
          key={customOpenCount}
          isOpen={customOpen}
          onOpenChange={setCustomOpen}
          onSubmit={priceCustom}
          collectModelKey
          title="Price a model"
          description={() =>
            "Set what a model costs by its selector, for a backend the catalog cannot list. Requests from now on are costed at these rates and counted against budgets."
          }
        />
      ) : null}
    </>
  )
}
