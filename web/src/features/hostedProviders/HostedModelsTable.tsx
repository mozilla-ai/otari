import { FiEdit2, FiRotateCcw } from "react-icons/fi"

import type { HostedModel } from "@/client"
import { RowAction, RowActionRow } from "@/design-system/actions/RowAction"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { Toggle } from "@/design-system/forms/Toggle"
import { Badge } from "@/design-system/indicators/Badge"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import { offeredPriceColumns } from "@/features/models/offeredPriceColumns"
import { formatOfferingPrice } from "@/features/models/pricingUnit"

/**
 * Where a model's rate comes from. `provider` is the price the provider's own
 * listing named, as another Otari's does. A default on an `otari` provider is
 * the vendor's public rate, which the Otari behind it may not charge, so it is
 * named an estimate.
 */
function PriceSourceBadge({
  source,
  isEstimate,
}: {
  source: HostedModel["price_source"]
  isEstimate: boolean
}) {
  if (source === "deployment") return <Badge tone="muted">Custom</Badge>
  if (source === "provider") return <Badge tone="muted">Listed</Badge>
  if (source === "defaults") {
    return <Badge tone="muted">{isEstimate ? "Estimate" : "Default"}</Badge>
  }
  return <Badge tone="warn">Unpriced</Badge>
}

/**
 * Models in today's row shape: price, where it comes from, the serving switch
 * and Edit price. A Custom price the default rung now prices differently names
 * that default under its badge and offers to go back to it.
 */
export function HostedModelsTable({
  ariaLabel,
  rows,
  isEstimate,
  isVendorOff,
  isLoading,
  emptyContent,
  pendingModelId,
  onToggle,
  onEdit,
  onUseDefault,
}: {
  ariaLabel: string
  rows: HostedModel[]
  /** The provider is another Otari, whose default is its listing or an estimate. */
  isEstimate: boolean
  /** The models' vendor is switched off, which holds every switch here. */
  isVendorOff: boolean
  isLoading: boolean
  emptyContent: string | undefined
  /** The model whose switch is being written, or empty. */
  pendingModelId: string
  onToggle: (row: HostedModel, enabled: boolean) => void
  onEdit: (row: HostedModel) => void
  onUseDefault: (row: HostedModel) => void
}) {
  const defaultName = isEstimate ? "Default" : "models.dev"
  const columns: DataTableColumn<HostedModel>[] = [
    {
      id: "model",
      header: "Model",
      isRowHeader: true,
      cell: (row) => (
        <span className={isVendorOff ? "text-subtle" : undefined}>
          {row.model}
        </span>
      ),
    },
    ...offeredPriceColumns<HostedModel>(),
    {
      id: "source",
      header: "Rate",
      cell: (row) => (
        <span className="flex flex-col items-start gap-0.5">
          <PriceSourceBadge source={row.price_source} isEstimate={isEstimate} />
          {row.default_rates ? (
            <span className="text-caption text-subtle tabular-nums">
              {defaultName} {formatOfferingPrice(row.default_rates)}
            </span>
          ) : null}
        </span>
      ),
    },
    {
      id: "enabled",
      header: "Serving",
      cell: (row) => (
        <div className="flex items-center gap-3">
          <Toggle
            isSelected={row.enabled}
            // Only the row being written: one in-flight toggle must not freeze
            // the rest of the table.
            isDisabled={isVendorOff || pendingModelId === row.id}
            label={`Serve ${row.model}`}
            onChange={(enabled) => onToggle(row, enabled)}
          />
          {row.disabled_reason === "unavailable" ? (
            <Badge tone="warn">Unavailable</Badge>
          ) : null}
        </div>
      ),
    },
    {
      id: "actions",
      header: "Actions",
      align: "end",
      cell: (row) => (
        <RowActionRow>
          {row.default_rates ? (
            <RowAction
              icon={FiRotateCcw}
              label="Use default price"
              ariaLabel={`Use the default price for ${row.model}`}
              onPress={() => onUseDefault(row)}
            />
          ) : null}
          <RowAction
            icon={FiEdit2}
            label="Edit price"
            ariaLabel={`Edit ${row.model} price`}
            onPress={() => onEdit(row)}
          />
        </RowActionRow>
      ),
    },
  ]
  return (
    <TableScrollFrame className="otari-offered-models-table">
      <DataTable
        ariaLabel={ariaLabel}
        mobileLayout="cards"
        columns={columns}
        rows={rows}
        getRowKey={(row) => row.id}
        isLoading={isLoading}
        emptyContent={emptyContent}
      />
    </TableScrollFrame>
  )
}
