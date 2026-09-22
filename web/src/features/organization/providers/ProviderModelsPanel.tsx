import { useState } from "react"
import {
  FiEdit2,
  FiPlus,
  FiRefreshCw,
  FiRotateCcw,
  FiSlash,
} from "react-icons/fi"

import type { OrgProviderKey, OrgProviderModel } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ConfirmRowAction } from "@/design-system/actions/ConfirmRowAction"
import { RowAction, RowActionRow } from "@/design-system/actions/RowAction"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { TablePagination } from "@/design-system/data/TablePagination"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { EmptyState } from "@/design-system/feedback/EmptyState"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { Toggle } from "@/design-system/forms/Toggle"
import { Badge } from "@/design-system/indicators/Badge"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import {
  useOrgProviderModels,
  useRefreshOrgProviderModelPricing,
  useRefreshOrgProviderModels,
  useSetOrgProviderModelEnabled,
  useWithdrawOrgProviderModel,
} from "@/shared/api/organizations"
import { useDeleteOrganizationPricing } from "@/shared/api/pricing"
import { formatRate } from "@/shared/helpers/format"

import { OfferModelDialog } from "./OfferModelDialog"
import { refreshOutcome } from "./refreshOutcome"

// One provider key's models, opened as its row's detail.
//
// Each row shows what the model costs this organization, which rung of the
// pricing ladder said so, and whether the runtime serves it. The switch is the
// row's off button: a model is withheld rather than withdrawn, so its row, its
// rate and its history stay where an admin can turn it back on. Withdrawing is
// the separate, rarer action, and it keeps the rate for the same reason.
//
// Two refreshes, because they ask different questions at different costs.
// Refresh models dials the provider, which can take the whole discovery timeout;
// Refresh pricing only re-reads the community dataset. An admin who wants the
// second should not wait on the first.
//
// No gate of its own: it renders inside a table a caller who cannot manage the
// organization never sees. `canEdit` is still a prop so the controls can be
// disabled rather than absent where that reads better, and so this is testable
// on its own.

/**
 * A rate cell. An absent rate is a dash, never a zero, which is a real price.
 *
 * One spelling for absent, and it is `undefined`: the wire has the field
 * optional *and* nullable, and a rate has no empty value to use instead, since
 * zero is a real price. The call sites convert at the boundary.
 */
function rate(value: number | undefined) {
  return value === undefined ? (
    <span className="text-subtle">—</span>
  ) : (
    <span className="text-mono-caption tabular-nums">{formatRate(value)}</span>
  )
}

// Typed rather than `string`, so the day a rung is renamed or added this stops
// compiling instead of quietly falling through to Unpriced. The spellings are
// the gateway's one `PriceSource` vocabulary, which the Models page names too.
function PriceSourceBadge({
  source,
}: {
  source: OrgProviderModel["price_source"]
}) {
  if (source === "organization") return <Badge tone="muted">Your rate</Badge>
  if (source === "deployment") return <Badge tone="muted">Deployment</Badge>
  if (source === "defaults") return <Badge tone="muted">Default</Badge>
  return <Badge tone="warn">Unpriced</Badge>
}

export function ProviderModelsPanel({
  providerKey,
  canEdit,
  onEditRate,
  page,
  pageSize,
  onPageChange,
  onPageSizeChange,
}: {
  providerKey: OrgProviderKey
  canEdit: boolean
  /** Opens the organization's rate editor on one model, on the page above. */
  onEditRate: (model: OrgProviderModel) => void
  /**
   * Which page of models is shown. Owned by the page above, which keeps it in
   * the URL: an expanded panel is worth sharing, and so is where you were in it.
   */
  page: number
  pageSize: number
  onPageChange: (page: number) => void
  onPageSizeChange: (size: number) => void
}) {
  const models = useOrgProviderModels(providerKey.id, page, pageSize)
  const setEnabled = useSetOrgProviderModelEnabled(providerKey.id)
  const withdraw = useWithdrawOrgProviderModel(providerKey.id)
  const clearRate = useDeleteOrganizationPricing()
  const refreshModels = useRefreshOrgProviderModels(providerKey.id)
  const refreshPricing = useRefreshOrgProviderModelPricing(providerKey.id)

  const [isOffering, setIsOffering] = useState(false)
  const [offerOpenCount, setOfferOpenCount] = useState(0)
  const [pendingWithdraw, setPendingWithdraw] = useState<OrgProviderModel>()

  // Fresh on every open and untouched through the exit, the rule the page above
  // follows for its own form.
  const openOffer = () => {
    setOfferOpenCount((n) => n + 1)
    setIsOffering(true)
  }

  const rows = models.data?.data ?? []
  // The last refresh's outcome, worth a sentence either way: silence after a
  // press reads as a button that did nothing. Cleared by the next change made
  // here, so it never describes a list that has moved on.
  const outcome = refreshModels.data
    ? refreshOutcome(refreshModels.data, "models")
    : refreshPricing.data
      ? refreshOutcome(refreshPricing.data, "pricing")
      : undefined
  const clearOutcome = () => {
    refreshModels.reset()
    refreshPricing.reset()
  }

  const columns: DataTableColumn<OrgProviderModel>[] = [
    {
      id: "model",
      header: "Model",
      isRowHeader: true,
      cell: (row) => row.model,
    },
    {
      id: "input",
      header: "Input / 1M",
      align: "end",
      cell: (row) => rate(row.input_price_per_million ?? undefined),
    },
    {
      id: "output",
      header: "Output / 1M",
      align: "end",
      cell: (row) => rate(row.output_price_per_million ?? undefined),
    },
    {
      id: "cache_read",
      header: "Cache read",
      align: "end",
      cell: (row) => rate(row.cache_read_price_per_million ?? undefined),
    },
    {
      id: "cache_write",
      header: "Cache write",
      align: "end",
      cell: (row) => rate(row.cache_write_price_per_million ?? undefined),
    },
    {
      id: "source",
      header: "Rate",
      cell: (row) => <PriceSourceBadge source={row.price_source} />,
    },
    {
      id: "enabled",
      header: "Serving",
      cell: (row) => (
        <Toggle
          isSelected={row.enabled}
          // Three reasons a switch is not yours to move. Only the row being
          // written, not every row: one in-flight toggle must not freeze the
          // rest of the table. And an unpriced model cannot be switched *on*,
          // which the server refuses too, so the control says so rather than
          // taking a press and answering with a banner. Switching one off stays
          // available, so a row that reached that state some other way can
          // still be withdrawn.
          isDisabled={
            !canEdit ||
            (setEnabled.isPending &&
              setEnabled.variables?.modelId === row.id) ||
            (!row.enabled && !row.price_source)
          }
          label={
            !row.enabled && !row.price_source
              ? `${row.model} has no rate yet, so it cannot be served`
              : `Serve ${row.model}`
          }
          onChange={(enabled) => {
            clearOutcome()
            setEnabled.mutate({ modelId: row.id, enabled })
          }}
        />
      ),
    },
    {
      id: "actions",
      header: "Actions",
      align: "end",
      cell: (row) =>
        canEdit ? (
          <RowActionRow>
            <RowAction
              onPress={() => onEditRate(row)}
              icon={FiEdit2}
              label="Set your rate"
              ariaLabel={`Set your rate for ${row.model}`}
            />
            {/* Only where there is one to clear. Removing the organization's own
                rate returns the model to whatever priced it before: the
                deployment's list, or the community default a later refresh
                re-seeds. */}
            {row.price_source === "organization" && row.pricing_id ? (
              <ConfirmRowAction
                icon={FiRotateCcw}
                label={`Use the default rate for ${row.model}`}
                confirmLabel="Use default"
                isPending={clearRate.isPending}
                onConfirm={() =>
                  clearRate.mutate(row.pricing_id as string, {
                    onSuccess: clearOutcome,
                  })
                }
              />
            ) : null}
            {/* Quiet at rest. The confirm dialog is where this action states
                its consequence, and a row wearing the danger ink before anyone
                has armed it spends the color on a row that is fine. */}
            <RowAction
              onPress={() => setPendingWithdraw(row)}
              icon={FiSlash}
              label="Stop offering"
              ariaLabel={`Stop offering ${row.model}`}
            />
          </RowActionRow>
        ) : null,
    },
  ]

  if (models.isPending && !models.data) {
    return <PageLoading label="Loading models…" />
  }

  return (
    <div className="flex flex-col gap-3 p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex min-w-0 flex-col">
          <h3 className="text-title">Models on {providerKey.name}</h3>
          <p className="text-caption">
            Priced in USD per million tokens, for this organization only. A
            model nothing prices is offered but not served, and stays off until
            it has a rate.
          </p>
        </div>
        {canEdit ? (
          <div className="flex flex-wrap items-center gap-2">
            <Button
              size="sm"
              variant="ghost"
              className="min-h-11 md:min-h-9"
              isPending={refreshModels.isPending}
              onPress={() => {
                clearOutcome()
                refreshModels.mutate()
              }}
            >
              <FiRefreshCw aria-hidden className="size-4" />
              Refresh models
            </Button>
            <Button
              size="sm"
              variant="ghost"
              className="min-h-11 md:min-h-9"
              isPending={refreshPricing.isPending}
              onPress={() => {
                clearOutcome()
                refreshPricing.mutate()
              }}
            >
              <FiRefreshCw aria-hidden className="size-4" />
              Refresh pricing
            </Button>
            <Button
              size="sm"
              variant="primary"
              className="min-h-11 md:min-h-9"
              onPress={openOffer}
            >
              <FiPlus aria-hidden className="size-4" />
              Add model
            </Button>
          </div>
        ) : null}
      </div>

      <ErrorBanner
        error={
          models.error ??
          setEnabled.error ??
          clearRate.error ??
          refreshModels.error ??
          refreshPricing.error
        }
      />
      {outcome ? (
        <p className="text-caption" role="status">
          {outcome}
        </p>
      ) : null}

      {/* A failed list is not an empty one: with a settled error, rendering
          "no models yet" would tell an admin their catalog is gone. */}
      {!models.error && rows.length === 0 ? (
        <EmptyState
          title="No models offered yet"
          description="Refresh to offer what this provider lists, or add a model by name for a backend that publishes no list."
          actionLabel={canEdit ? "Refresh models" : undefined}
          onAction={canEdit ? () => refreshModels.mutate() : undefined}
          isActionDisabled={refreshModels.isPending}
        />
      ) : (
        <>
          <TableScrollFrame className="otari-offered-models-table">
            <DataTable
              ariaLabel={`Models on ${providerKey.name}`}
              columns={columns}
              rows={rows}
              getRowKey={(row) => row.id}
              isLoading={models.isLoading}
            />
          </TableScrollFrame>
          <TablePagination
            page={page}
            pageSize={pageSize}
            total={models.data?.count ?? null}
            rowsOnPage={rows.length}
            onPageChange={onPageChange}
            onPageSizeChange={onPageSizeChange}
            isFetching={models.isFetching}
            // Lower case and plural, which is this prop's contract: it is
            // suffixed onto the control names, not used as a heading.
            label={`models on ${providerKey.name}`}
          />
        </>
      )}

      <OfferModelDialog
        key={offerOpenCount}
        providerKey={providerKey}
        isOpen={isOffering}
        onOpenChange={setIsOffering}
        onOffered={() => {
          clearOutcome()
          setIsOffering(false)
        }}
      />

      <ConfirmDialog
        isOpen={pendingWithdraw !== undefined}
        onOpenChange={(open) => {
          if (open) return
          setPendingWithdraw(undefined)
          withdraw.reset()
        }}
        heading="Stop offering this model?"
        body={
          pendingWithdraw
            ? `${pendingWithdraw.model} leaves this organization's catalog. Its rate and its history stay, so offering it again finds the rate where you left it. To withhold it temporarily, use the Serving switch instead.`
            : null
        }
        confirmLabel="Stop offering"
        isPending={withdraw.isPending}
        error={withdraw.error}
        onConfirm={() => {
          if (!pendingWithdraw) return
          withdraw.mutate(pendingWithdraw.id, {
            onSuccess: () => setPendingWithdraw(undefined),
          })
        }}
      />
    </div>
  )
}
