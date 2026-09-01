import { AlertDialog, Button } from "@heroui/react"
import { Link } from "@tanstack/react-router"

import type { PricingRefreshPreview, PricingResponse } from "@/client"
import { currentPricing } from "@/features/models/pricing"
// Feature-to-feature, which the boundary rules allow: the overrides are the
// organization's own rates above this catalog, so they belong on this page while
// the tenancy feature keeps owning them.
import { RateOverridesCard } from "@/features/organization/RateOverridesCard"
import {
  useConfirmPricingRefresh,
  usePreviewPricingRefresh,
  usePricing,
  useRejectPricingRefresh,
  useSettings,
} from "@/shared/api/hooks"
import { DataTable, type DataTableColumn } from "@/shared/components/DataTable"
import {
  PageIntro,
  Section,
  TableScrollFrame,
} from "@/shared/components/surface"
import { ErrorBanner, InfoBanner, PageLoading } from "@/shared/components/ui"
import { formatCost, formatRelative } from "@/shared/helpers/format"

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
// The split it settles: **this page owns the catalog** (whether unpriced models
// are metered at all, where the defaults come from, and which models carry a
// custom rate), and **Models still owns one model's price**, because that is
// edited next to the model it applies to and reached from three places that all
// start with a specific model. So the table here links there rather than
// growing a second copy of that editor.

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
              <ul className="max-h-60 list-disc overflow-auto pl-5 text-sm text-foreground">
                {preview.changes.map((change) => (
                  <li key={change.model_key}>
                    {change.model_key}: {change.change}
                  </li>
                ))}
              </ul>
            ) : null}
            {preview.changes_truncated ? (
              <p className="text-xs text-muted">
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

function PricingRefreshSection() {
  const previewRefresh = usePreviewPricingRefresh()
  const confirmRefresh = useConfirmPricingRefresh()
  const rejectRefresh = useRejectPricingRefresh()
  const preview = previewRefresh.data
  const isPending = confirmRefresh.isPending || rejectRefresh.isPending

  const reject = () => {
    if (preview === undefined || isPending) {
      return
    }
    rejectRefresh.mutate(undefined, { onSuccess: previewRefresh.reset })
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
          </div>
          <Button
            size="sm"
            variant="outline"
            isDisabled={previewRefresh.isPending || isPending}
            onPress={() => previewRefresh.mutate()}
          >
            {previewRefresh.isPending
              ? "Checking prices…"
              : "Check for price updates"}
          </Button>
        </div>
        <ErrorBanner error={previewRefresh.error} />
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
                onSuccess: previewRefresh.reset,
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
  updatedAt: string
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
function currentRows(all: PricingResponse[]): PriceRow[] {
  return currentPricing(all).map((live) => ({
    modelKey: live.model_key,
    input: live.input_price_per_million,
    output: live.output_price_per_million,
    cacheRead: live.cache_read_price_per_million,
    tiers: live.pricing_tiers.length,
    updatedAt: live.updated_at,
  }))
}

const COLUMNS: DataTableColumn<PriceRow>[] = [
  {
    id: "modelKey",
    header: "Model",
    isRowHeader: true,
    cell: (row) => (
      <span className="font-mono text-sm text-foreground">{row.modelKey}</span>
    ),
  },
  {
    id: "input",
    header: "Input / 1M",
    align: "end",
    cell: (row) => formatCost(row.input),
  },
  {
    id: "output",
    header: "Output / 1M",
    align: "end",
    cell: (row) => formatCost(row.output),
  },
  {
    id: "cacheRead",
    header: "Cache read / 1M",
    align: "end",
    // An em dash rather than $0.00: a model with no cache-read rate is not the
    // same as one that reads cache for free.
    cell: (row) => (row.cacheRead === null ? "—" : formatCost(row.cacheRead)),
  },
  {
    id: "tiers",
    header: "Tiers",
    align: "end",
    cell: (row) => (row.tiers === 0 ? "—" : `${row.tiers} configured`),
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

function PriceTable() {
  const pricing = usePricing()
  const rows = pricing.data ? currentRows(pricing.data) : []

  if (pricing.isLoading) return <PageLoading label="Loading model prices…" />

  return (
    <>
      {/* The group's heading and the rule under it are what introduce the rows,
          which then sit straight on the page ground. No box: the header rule
          and the row separators already say where the group starts and ends. */}
      <Section className="pt-6 pb-3">
        <h2 className="text-title">Model prices</h2>
      </Section>
      <ErrorBanner error={pricing.error} />
      <TableScrollFrame className="otari-pricing-table">
        <DataTable
          ariaLabel="Model prices"
          columns={COLUMNS}
          rows={rows}
          getRowKey={(row) => row.modelKey}
          emptyContent="No model carries a stored price yet. Price one from the Models page."
        />
      </TableScrollFrame>
      <Section className="pt-3">
        <p className="text-sm text-muted">
          A rate is edited beside the model it applies to, on{" "}
          <Link
            to="/models"
            className="font-medium text-link hover:text-link-hover"
          >
            Models
          </Link>
          .
        </p>
      </Section>
    </>
  )
}

export function ModelPricingPage() {
  return (
    <div className="flex flex-col">
      <PageIntro title="Model pricing">
        What this gateway meters a request at. The catalog applies to every
        workspace and every key in the organization; a rate override below
        applies to this organization ahead of it.
      </PageIntro>
      <CatalogPolicy />
      <PricingRefreshSection />
      <PriceTable />
      <RateOverridesCard />
    </div>
  )
}
