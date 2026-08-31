import { AlertDialog, Button, Card } from "@heroui/react"
import { Link } from "@tanstack/react-router"

import type { PricingRefreshPreview, PricingResponse } from "@/client"
import { currentPricing } from "@/features/models/pricing"
// Feature-to-feature, which the boundary rules allow: the overrides are the
// organization's own rates above this catalog, so they belong on this page while
// the tenancy feature keeps owning them.
import { RateOverridesCard } from "@/features/organization/RateOverridesCard"
import { isDeploymentOperator } from "@/features/organization/roles"
import {
  useConfirmPricingRefresh,
  useOrganizationContext,
  usePreviewPricingRefresh,
  usePricing,
  useRejectPricingRefresh,
  useSettings,
} from "@/shared/api/hooks"
import { DataTable, type DataTableColumn } from "@/shared/components/DataTable"
import {
  ErrorBanner,
  InfoBanner,
  PageHeader,
  PageLoading,
} from "@/shared/components/ui"
import { formatCost } from "@/shared/helpers/format"

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
    <section className="flex flex-col gap-2">
      <h2 className="text-title">Default pricing catalog</h2>
      <Card>
        <Card.Content className="flex flex-col gap-4 p-5">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="min-w-0">
              <div className="text-sm font-medium text-foreground">
                genai-prices defaults
              </div>
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
        </Card.Content>
      </Card>
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
    </section>
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
  const rows = pricing.data ? currentRows(pricing.data) : []

  if (pricing.isLoading) return <PageLoading label="Loading model prices…" />

  return (
    <section className="flex flex-col gap-2">
      <h2 className="text-title">Model prices</h2>
      <ErrorBanner error={pricing.error} />
      <Card>
        <Card.Content className="p-0">
          <DataTable
            ariaLabel="Model prices"
            columns={COLUMNS}
            rows={rows}
            getRowKey={(row) => row.modelKey}
            emptyContent={
              canPrice
                ? "No model carries a stored price yet. Price one from the Models page."
                : "No model carries a stored price yet."
            }
          />
        </Card.Content>
      </Card>
      {canPrice ? (
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
      ) : null}
    </section>
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
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Model pricing"
        description="What this gateway meters a request at. The catalog applies to every workspace and every key in the organization; a rate override below applies to this organization ahead of it."
      />
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
