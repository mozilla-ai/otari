import { useState } from "react"
import { FiEdit2, FiPlus, FiRefreshCw } from "react-icons/fi"

import type { HostedModel, HostedProvider } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { RowAction, RowActionRow } from "@/design-system/actions/RowAction"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import {
  PAGE_SIZE_OPTIONS,
  TablePagination,
} from "@/design-system/data/TablePagination"
import { EmptyState } from "@/design-system/feedback/EmptyState"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { Toggle } from "@/design-system/forms/Toggle"
import { Badge } from "@/design-system/indicators/Badge"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import { formatPricedRate, pricingUnitOf } from "@/features/models/pricingUnit"
import {
  useHostedModels,
  useRefreshHostedModels,
  useUpdateHostedModel,
} from "@/shared/api/hostedProviders"
import { formatRate } from "@/shared/helpers/format"

import { HostedModelForm } from "./HostedModelForm"
import { refreshOutcome } from "./refreshOutcome"

// One hosted provider's offered models.
//
// Each row shows the price the model serves at, which rung set it, and a
// Serving switch. Custom is a rate an operator set; Default is the community
// rate, which Refresh keeps current until someone sets one; Unpriced is a model
// nothing prices, which arrives switched off. The switch is the row's off
// button: a model is withheld rather than removed, so its row, its price and its
// history stay where the operator can turn it back on.

/** A rate cell. An absent rate is a dash, never a zero, which is a real price. */
function rate(value: number | null | undefined) {
  return value === null || value === undefined ? (
    <span className="text-subtle">—</span>
  ) : (
    <span className="text-mono-caption tabular-nums">{formatRate(value)}</span>
  )
}

/**
 * The input cell, which names its unit when the rate is not per token: a
 * per-request rate of 2000 per million under "Input / 1M" would read as a
 * token price a thousand times too high.
 */
function inputRate(row: HostedModel) {
  const unit = pricingUnitOf(row.unit)
  const value = row.input_price_per_million
  if (unit === "tokens" || value === null || value === undefined) {
    return rate(value)
  }
  return (
    <span className="text-mono-caption tabular-nums">
      {formatPricedRate(value, unit)}
    </span>
  )
}

/** A rate that only a token price carries, as a dash on any other unit. */
function tokenRate(row: HostedModel, value: number | null | undefined) {
  return rate(pricingUnitOf(row.unit) === "tokens" ? value : undefined)
}

function PriceSourceBadge({ source }: { source: HostedModel["price_source"] }) {
  if (source === "deployment") return <Badge tone="muted">Custom</Badge>
  if (source === "defaults") return <Badge tone="muted">Default</Badge>
  return <Badge tone="warn">Unpriced</Badge>
}

export function HostedProviderModelsPanel({
  provider,
}: {
  provider: HostedProvider
}) {
  const [page, setPage] = useState(0)
  const [pageSize, setPageSize] = useState<number>(PAGE_SIZE_OPTIONS[0])
  const models = useHostedModels(provider.name, page, pageSize)
  const update = useUpdateHostedModel(provider.name)
  const refresh = useRefreshHostedModels(provider.name)

  const [isAdding, setIsAdding] = useState(false)
  const [addOpenCount, setAddOpenCount] = useState(0)
  const [editingId, setEditingId] = useState("")

  // Fresh on every open: the create form is keyed on the open count.
  const openAdd = () => {
    setEditingId("")
    setAddOpenCount((n) => n + 1)
    setIsAdding(true)
  }

  const rows = models.data?.data ?? []
  const editing = rows.find((row) => row.id === editingId)
  // The last refresh's outcome, worth a sentence either way: silence after a
  // press reads as a button that did nothing. Cleared by the next change made
  // here, so it never describes a list that has moved on.
  const outcome = refresh.data ? refreshOutcome(refresh.data) : undefined

  const columns: DataTableColumn<HostedModel>[] = [
    {
      id: "model",
      header: "Model",
      isRowHeader: true,
      cell: (row) => row.model,
    },
    { id: "input", header: "Input / 1M", align: "end", cell: inputRate },
    {
      id: "output",
      header: "Output / 1M",
      align: "end",
      cell: (row) => tokenRate(row, row.output_price_per_million),
    },
    {
      id: "cache_read",
      header: "Cache read",
      align: "end",
      cell: (row) => tokenRate(row, row.cache_read_price_per_million),
    },
    {
      id: "cache_write",
      header: "Cache write",
      align: "end",
      cell: (row) => tokenRate(row, row.cache_write_price_per_million),
    },
    {
      id: "source",
      header: "Price",
      cell: (row) => <PriceSourceBadge source={row.price_source} />,
    },
    {
      id: "enabled",
      header: "Serving",
      cell: (row) => (
        <Toggle
          isSelected={row.enabled}
          // Only the row being written: one in-flight toggle must not freeze
          // the rest of the table.
          isDisabled={update.isPending && update.variables?.modelId === row.id}
          label={`Serve ${row.model}`}
          onChange={(enabled) => {
            refresh.reset()
            update.mutate({ modelId: row.id, body: { enabled } })
          }}
        />
      ),
    },
    {
      id: "actions",
      header: "Actions",
      align: "end",
      cell: (row) => (
        <RowActionRow>
          <RowAction
            icon={FiEdit2}
            label="Edit price"
            ariaLabel={`Edit ${row.model} price`}
            onPress={() => {
              setIsAdding(false)
              setEditingId(row.id)
            }}
          />
        </RowActionRow>
      ),
    },
  ]

  if (models.isPending && !models.data) {
    return <PageLoading label="Loading models…" />
  }

  return (
    <div className="flex flex-col gap-3 p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex min-w-0 flex-col">
          <h3 className="text-title">Models on {provider.name}</h3>
          {provider.source === "config" ? (
            <p className="text-caption">
              This provider is defined in config.yml: change its key, endpoint
              or switch there. Its models are managed here.
            </p>
          ) : null}
          <p className="text-caption">
            Priced in USD per million tokens unless a rate says it is per
            request or per image. A model offered without a price starts
            switched off, and stays off until it is priced or switched on.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Button
            size="sm"
            className="min-h-11 md:min-h-9"
            isPending={refresh.isPending}
            onPress={() => refresh.mutate()}
          >
            <FiRefreshCw aria-hidden className="size-4" />
            Refresh models
          </Button>
          <Button
            size="sm"
            variant="primary"
            className="min-h-11 md:min-h-9"
            onPress={openAdd}
          >
            <FiPlus aria-hidden className="size-4" />
            Add model
          </Button>
        </div>
      </div>

      <ErrorBanner error={models.error ?? update.error ?? refresh.error} />
      {outcome ? (
        <p className="text-caption" role="status">
          {outcome}
        </p>
      ) : null}

      {/* A failed list is not an empty one. */}
      {!models.error && rows.length === 0 ? (
        <EmptyState
          title="No models offered yet"
          description="Refresh to offer what the provider lists, or add a model to say what this provider serves and at what price."
          actionLabel="Refresh models"
          onAction={() => refresh.mutate()}
          isActionDisabled={refresh.isPending}
        />
      ) : (
        <>
          <TableScrollFrame className="otari-offered-models-table">
            <DataTable
              ariaLabel={`Models on ${provider.name}`}
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
            onPageChange={setPage}
            onPageSizeChange={(size) => {
              setPageSize(size)
              setPage(0)
            }}
            isFetching={models.isFetching}
            label={`models on ${provider.name}`}
          />
        </>
      )}

      <HostedModelForm
        key={addOpenCount}
        provider={provider}
        editing={undefined}
        isOpen={isAdding}
        onClose={() => setIsAdding(false)}
        onSaved={() => refresh.reset()}
      />
      {editing ? (
        // Remounted per row, so a second row's editor cannot open with the
        // first one's draft.
        <HostedModelForm
          key={editing.id}
          provider={provider}
          editing={editing}
          isOpen
          onClose={() => setEditingId("")}
          onSaved={() => refresh.reset()}
        />
      ) : null}
    </div>
  )
}
