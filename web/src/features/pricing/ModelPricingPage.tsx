import { AlertDialog, Button } from "@heroui/react"
import { Link, useNavigate } from "@tanstack/react-router"
import { useState } from "react"

import type {
  PricingDriftRow,
  PricingRefreshPreview,
  PricingResponse,
} from "@/client"
import { currentPricing } from "@/features/models/pricing"
import {
  type ManualRates,
  SetPriceDialog,
} from "@/features/models/SetPriceDialog"
// Feature-to-feature, which the boundary rules allow: the overrides are the
// organization's own rates above this catalog, so they belong on this page while
// the tenancy feature keeps owning them.
import { RateOverridesCard } from "@/features/organization/RateOverridesCard"
import { isDeploymentOperator } from "@/features/organization/roles"
import { PriceEditor } from "@/features/pricing/PriceEditor"
import { useOrganizationContext } from "@/shared/api/organizations"
import {
  useConfirmPricingRefresh,
  usePendingPricingRefresh,
  usePreviewPricingRefresh,
  usePricing,
  usePricingDrift,
  usePricingSnapshots,
  useRejectPricingRefresh,
  useSetPricing,
} from "@/shared/api/pricing"
import { useSettings } from "@/shared/api/settings"
import {
  DataTable,
  type DataTableColumn,
} from "@/shared/components/data/DataTable"
import { ErrorBanner } from "@/shared/components/feedback/ErrorBanner"
import { InfoBanner } from "@/shared/components/feedback/InfoBanner"
import { PageLoading } from "@/shared/components/feedback/PageLoading"
import { PageIntro } from "@/shared/components/layout/PageIntro"
import { Section } from "@/shared/components/layout/Section"
import { TableScrollFrame } from "@/shared/components/layout/TableScrollFrame"
import { formatRate, formatRelative } from "@/shared/helpers/format"
import { useUrlValue } from "@/shared/helpers/urlState"

// The organization's model pricing: what the gateway meters a request at, and
// where the numbers come from.
//
// The navigation design gives this a destination of its own on the organization
// rail, under Cost & billing, and pricing is genuinely tenant-scoped: a rate
// applies to every workspace and every key in the deployment. It had no home
// before: the default catalog's refresh flow was buried in the gateway's runtime
// Settings page, next to the master key, and the per-model rates were a column
// on Models.
//
// The split it settles: **this page owns pricing**, the policy, the defaults,
// the stored rates and the editor for one of them. Models is read-only for
// everyone (otari-ai#2095, #2096): its detail links here with the selector in
// `?model=`, which opens the editor below for an operator, so the page that
// compares prices is never the page that changes them.
//
// The other split, which is about who is asking (otari-ai#1943): the page holds
// a deployment-wide half and a tenant-scoped half, and the roles matrix puts it
// at Edit for an organization admin. So the halves are gated separately rather
// than the destination being operator-only:
//
// - The **catalog policy** and the **refresh flow** are the deployment's. Both
//   are `require_deployment_operator` server-side (`GET /v1/settings`, and the
//   three `/v1/pricing/refresh` routes), so they are withheld from anyone else
//   rather than fired into a 403 banner, the way `ModelsPage` withholds its own
//   operator-only reads.
// - The **price table** reads `/v1/pricing`, which `verify_catalog_reader`
//   already serves to any session.
// - The **rate overrides** are the organization's own, and
//   `organization_pricing_service` gates the writes on the same owner-or-admin
//   role the card already asks about, so it needs nothing here.

function PricingRefreshDialog({
  preview,
  error,
  isPending,
  onAccept,
  onReject,
}: {
  preview: PricingRefreshPreview
  error: Error | null
  isPending: boolean
  onAccept: () => void
  onReject: () => void
}) {
  return (
    <AlertDialog.Backdrop>
      <AlertDialog.Container placement="center" size="lg">
        <AlertDialog.Dialog>
          <AlertDialog.Header>
            <AlertDialog.Heading>
              Review default price updates
            </AlertDialog.Heading>
          </AlertDialog.Header>
          <AlertDialog.Body className="flex flex-col gap-4">
            <p className="text-sm text-muted">
              {preview.added_count} added, {preview.changed_count} changed, and{" "}
              {preview.removed_count} removed upstream model prices. The
              accepted catalog is saved in the database with source{" "}
              <code>genai-prices</code> and reloads after a restart. Your{" "}
              {preview.protected_model_count} custom model price
              {preview.protected_model_count === 1 ? "" : "s"} remain unchanged.
            </p>
            {preview.changes.length > 0 ? (
              <ul className="max-h-60 list-disc overflow-auto pl-5 text-body">
                {preview.changes.map((change) => (
                  <li key={change.model_key}>
                    {change.model_key}: {change.change}
                  </li>
                ))}
              </ul>
            ) : null}
            {preview.changes_truncated ? (
              <p className="text-caption">
                Only the first 100 changes are shown.
              </p>
            ) : null}
            <ErrorBanner error={error} />
          </AlertDialog.Body>
          <AlertDialog.Footer>
            <Button variant="ghost" isDisabled={isPending} onPress={onReject}>
              Reject changes
            </Button>
            <Button variant="primary" isPending={isPending} onPress={onAccept}>
              Accept price updates
            </Button>
          </AlertDialog.Footer>
        </AlertDialog.Dialog>
      </AlertDialog.Container>
    </AlertDialog.Backdrop>
  )
}

// Who accepted a snapshot, as a word. The schedule is the only non-person.
function acceptedBy(who: string): string {
  return who === "schedule" ? "the scheduled check" : `an ${who}`
}

function PricingRefreshSection() {
  const previewRefresh = usePreviewPricingRefresh()
  const confirmRefresh = useConfirmPricingRefresh()
  const rejectRefresh = useRejectPricingRefresh()
  // What the scheduled check left for review under `pricing_refresh: review`.
  // It is the same pending row a manual check writes, so the one dialog and
  // the same confirm and reject serve both; the only difference is who fetched.
  const pending = usePendingPricingRefresh()
  const snapshots = usePricingSnapshots()
  const [reviewingPending, setReviewingPending] = useState(false)
  const preview =
    previewRefresh.data ??
    (reviewingPending ? pending.data : undefined) ??
    undefined
  const isPending = confirmRefresh.isPending || rejectRefresh.isPending
  const latest = snapshots.data?.[0]

  const close = () => {
    previewRefresh.reset()
    setReviewingPending(false)
  }
  const reject = () => {
    if (preview === undefined || isPending) {
      return
    }
    rejectRefresh.mutate(undefined, { onSuccess: close })
  }

  return (
    <>
      <Section
        aria-labelledby="pricing-catalog-title"
        className="border-y border-border py-5"
        contentClassName="flex flex-col gap-4"
      >
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            <h2 id="pricing-catalog-title" className="text-title">
              Default pricing catalog
            </h2>
            <p className="mt-1 max-w-3xl text-sm text-muted">
              Fetch the latest upstream catalog, review the proposed change
              summary, then accept or reject it. Accepted data is stored as{" "}
              <code>genai-prices</code>; custom prices remain separate and
              always take precedence.
            </p>
            {latest ? (
              <p className="mt-1 text-caption">
                Last accepted {formatRelative(latest.accepted_at)} by{" "}
                {acceptedBy(latest.accepted_by)}, {latest.model_count} priced
                models.
                {snapshots.data && snapshots.data.length > 1
                  ? ` ${snapshots.data.length} snapshots on record.`
                  : ""}
              </p>
            ) : null}
          </div>
          <Button
            size="sm"
            variant="ghost"
            isDisabled={previewRefresh.isPending || isPending}
            onPress={() => previewRefresh.mutate()}
          >
            {previewRefresh.isPending
              ? "Checking prices…"
              : "Check for price updates"}
          </Button>
        </div>
        <ErrorBanner error={previewRefresh.error} />
        {pending.data ? (
          <div className="flex flex-wrap items-center justify-between gap-3">
            <InfoBanner tone="warning">
              The scheduled check found {pending.data.changed_count} changed,{" "}
              {pending.data.added_count} added and {pending.data.removed_count}{" "}
              removed default prices, fetched{" "}
              {formatRelative(pending.data.fetched_at)}. Nothing changes until
              you accept.
            </InfoBanner>
            <Button
              size="sm"
              variant="primary"
              isDisabled={isPending}
              onPress={() => setReviewingPending(true)}
            >
              Review pending update
            </Button>
          </div>
        ) : null}
      </Section>
      <AlertDialog
        isOpen={preview !== undefined}
        onOpenChange={(isOpen) => (!isOpen ? reject() : undefined)}
      >
        <AlertDialog.Trigger className="hidden">
          Review price updates
        </AlertDialog.Trigger>
        {preview ? (
          <PricingRefreshDialog
            preview={preview}
            error={confirmRefresh.error ?? rejectRefresh.error}
            isPending={isPending}
            onAccept={() =>
              confirmRefresh.mutate(undefined, {
                onSuccess: close,
              })
            }
            onReject={reject}
          />
        ) : null}
      </AlertDialog>
    </>
  )
}

/**
 * Whether an unpriced model is metered at all, which is the catalog's first
 * question and the one that decides what the table below is for.
 *
 * With default pricing on, the table is the exceptions: models whose stored rate
 * overrides an upstream default. With it off, the table is the whole of what can
 * be billed, and `require_pricing` decides whether anything else is refused
 * outright or served for free.
 *
 * Mounted only for a deployment operator, which is why it reads `useSettings()`
 * with no gate of its own: `GET /v1/settings` is operator-only, so an admin who
 * saw this banner would be reading a refusal. Unmounted rather than passed a
 * disabled query, so nothing is left holding a cached answer from a session that
 * used to be an operator's.
 */
function CatalogPolicy() {
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

interface PriceRow {
  modelKey: string
  input: number
  output: number
  cacheRead: number | null
  tiers: number
  unit: string
  updatedAt: string
  /** Where this rate came from: `config`, `api`, or absent for a row older than the column. */
  origin: string | null
  /** How far the rate sits from today's genai-prices default, where one exists. */
  drift?: PricingDriftRow
}

/**
 * One row per priced model, from the price that is in force today.
 *
 * `/v1/pricing` returns the history, not the current state: a model repriced
 * three times has three rows, and only the newest one whose `effective_at` has
 * passed is what a request is metered at. `currentPricing` is the reduction
 * Models already uses, sorting included, so the two pages cannot disagree about
 * which rate is live.
 */
function currentRows(
  all: PricingResponse[],
  drift: readonly PricingDriftRow[] = [],
): PriceRow[] {
  const byKey = new Map(drift.map((row) => [row.model_key, row]))
  return currentPricing(all).map((live) => ({
    modelKey: live.model_key,
    input: live.input_price_per_million,
    output: live.output_price_per_million,
    cacheRead: live.cache_read_price_per_million,
    tiers: live.pricing_tiers.length,
    unit: live.unit,
    updatedAt: live.updated_at,
    origin: live.origin ?? null,
    drift: byKey.get(live.model_key),
  }))
}

/** A signed percentage, or the dash for a rate with nothing to compare to. */
export function formatDrift(delta: number | null | undefined): string {
  if (delta == null) return "—"
  const rounded = Math.round(delta)
  if (rounded === 0) return "±0%"
  return `${rounded > 0 ? "+" : "−"}${Math.abs(rounded)}%`
}

// Beyond this the stored rate is more than a rounding away from the default,
// and the cell says so in the danger ink.
const DRIFT_NOTICE_PERCENT = 10

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

// What a row's rates are per. A tool's row is per million requests, and the
// column heads say "/ 1M", so the unit is the lane that keeps that honest.
const UNIT_LABELS: Record<string, string> = {
  tokens: "tokens",
  requests: "requests",
  images: "images",
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
function PriceTable({ canPrice }: { canPrice: boolean }) {
  const pricing = usePricing()
  // Operator-only read, so it is gated on the same axis as the editor.
  const drift = usePricingDrift(canPrice)
  const navigate = useNavigate()
  const setPricing = useSetPricing()
  // The selector whose deployment rate is being edited, carried in the URL so
  // the catalog's "Edit rate" link lands here with it in hand.
  const editingKey = useUrlValue("model")
  const [customOpen, setCustomOpen] = useState(false)
  const rows = pricing.data ? currentRows(pricing.data, drift.data ?? []) : []
  const current = pricing.data
    ? currentPricing(pricing.data).find((row) => row.model_key === editingKey)
    : undefined

  const edit = (modelKey: string | null) =>
    void navigate({
      to: "/organization/pricing",
      search: modelKey ? { model: modelKey } : {},
    })

  // A backend with no /v1/models endpoint serves models the catalog never
  // lists, so the only way to meter them is a key typed by hand. The stored key
  // is what the server normalized, so the editor opens on that rather than on
  // the raw input.
  const priceCustom = (rates: ManualRates, modelKey: string) => {
    setPricing.mutate(
      {
        model_key: modelKey,
        input_price_per_million: rates.input_price_per_million,
        output_price_per_million: rates.output_price_per_million,
        cache_read_price_per_million:
          rates.cache_read_price_per_million ?? null,
        cache_write_price_per_million:
          rates.cache_write_price_per_million ?? null,
      },
      {
        onSuccess: (created) => {
          setCustomOpen(false)
          edit(created.model_key)
        },
      },
    )
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
          <Button size="sm" variant="ghost" onPress={() => setCustomOpen(true)}>
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
      {canPrice ? (
        <SetPriceDialog
          isOpen={customOpen}
          onOpenChange={setCustomOpen}
          isPending={setPricing.isPending}
          error={setPricing.error}
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

export function ModelPricingPage() {
  // The caller axis, read once for the whole page. Withholding the two
  // deployment-wide sections is not a second opinion about authorization: the
  // server refuses those reads to a non-operator either way, and this is only
  // what keeps an admin's own page from being three quarters refusal banner
  // (the shape otari#838 removed from the members roster).
  const organization = useOrganizationContext()
  const isOperator = isDeploymentOperator(organization.data)
  return (
    <div className="flex flex-col">
      <PageIntro title="Model pricing">
        What this gateway meters a request at. The catalog applies to every
        workspace and every key in the organization; a rate override below
        applies to this organization ahead of it.
      </PageIntro>
      {/* Operator-only: both read `/v1/pricing`'s catalog controls, which an
          organization admin may see prices through but not administer. */}
      {isOperator ? (
        <>
          <CatalogPolicy />
          <PricingRefreshSection />
        </>
      ) : null}
      <PriceTable canPrice={isOperator} />
      <RateOverridesCard />
    </div>
  )
}
