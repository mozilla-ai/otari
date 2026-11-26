import { FiEdit2, FiRotateCcw } from "react-icons/fi"

import type { HostedModel, HostedProviderRegion } from "@/client"
import { RowAction, RowActionRow } from "@/design-system/actions/RowAction"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { Toggle } from "@/design-system/forms/Toggle"
import { Badge } from "@/design-system/indicators/Badge"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import { Tooltip } from "@/design-system/overlays/Tooltip"
import { offeredPriceColumns } from "@/features/models/offeredPriceColumns"
import {
  formatOfferingPrice,
  pricingUnitOf,
} from "@/features/models/pricingUnit"
import { formatRate } from "@/shared/helpers/format"

import { regionRatesOf } from "./modelPriceDraft"

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

/** A price as the lines a narrow cell stacks: input over output for tokens, else its own unit. */
function priceLines(
  rates: {
    input_price_per_million: number
    output_price_per_million: number
    unit?: string | null
  } | null,
): string[] {
  if (rates === null) return ["Unpriced"]
  if (pricingUnitOf(rates.unit) !== "token") return [formatOfferingPrice(rates)]
  return [
    formatRate(rates.input_price_per_million),
    formatRate(rates.output_price_per_million),
  ]
}

/**
 * One region's price of a model, input over output. A price the region sets
 * of its own is in the body ink; one it does not set is the provider's, which
 * is what the region charges, so it is muted and italic. The whole cell opens
 * that region's editor.
 */
function RegionPriceCell({
  row,
  region,
  onEdit,
}: {
  row: HostedModel
  region: HostedProviderRegion
  onEdit: () => void
}) {
  const own = regionRatesOf(row, region.slug)
  const inherited =
    row.input_price_per_million == null || row.output_price_per_million == null
      ? null
      : {
          input_price_per_million: row.input_price_per_million,
          output_price_per_million: row.output_price_per_million,
          unit: row.unit,
        }
  const lines = priceLines(own ?? inherited)
  const price = formatOfferingPrice(own ?? inherited)
  return (
    // A bare button rather than `Button`: the whole cell is the target, and a
    // bordered button per region would neither fit nor let the ink say whose
    // price it is.
    <button
      type="button"
      aria-label={`Edit ${row.model} price in ${region.label}: ${price}${own ? "" : ", the provider's price"}`}
      title={
        own ? undefined : "The provider's price, which this region charges"
      }
      onClick={onEdit}
      className={`flex min-h-11 w-full cursor-pointer flex-col items-end justify-center px-1 text-right text-mono-caption whitespace-nowrap tabular-nums outline-none hover:bg-surface-alt focus-visible:otari-focus-ring ${own ? "font-medium text-foreground" : "text-subtle italic"}`}
    >
      <span>{lines[0]}</span>
      {lines[1] ? <span>{lines[1]}</span> : null}
    </button>
  )
}

/** The provider price's header beside regions: the default they fall back on. */
function defaultHeader(label: string) {
  return (
    <Tooltip
      content="What a region without a price of its own charges. $ per million tokens for text models."
      delay={0}
    >
      <span className="underline decoration-dotted underline-offset-4">
        {label}
      </span>
    </Tooltip>
  )
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
  regions = [],
  hasRegions = regions.length > 0,
  onToggle,
  onEdit,
  onEditRegionPrice,
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
  /**
   * The regions that get a price column each, after the provider's price. The
   * caller passes them only for a provider with two or more.
   */
  regions?: readonly HostedProviderRegion[]
  /**
   * The provider has regions, whether or not any has a column here: its own
   * price is then the default for regions without one.
   */
  hasRegions?: boolean
  onEditRegionPrice?: (row: HostedModel, region: HostedProviderRegion) => void
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
    // Beside region columns the provider's cache rates give way, so a row
    // still fits a desktop width; the price editor carries them. Its input and
    // output are then the default every region without a price charges.
    ...(hasRegions
      ? offeredPriceColumns<HostedModel>()
          .filter(
            (column) =>
              column.id !== "cache_read" && column.id !== "cache_write",
          )
          .map((column) => ({
            ...column,
            header: defaultHeader(
              column.id === "input" ? "Default in" : "Default out",
            ),
          }))
      : offeredPriceColumns<HostedModel>()),
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
    ...regions.map(
      (region): DataTableColumn<HostedModel> => ({
        id: `region:${region.slug}`,
        header: region.label,
        align: "end",
        cell: (row) => (
          <RegionPriceCell
            row={row}
            region={region}
            onEdit={() => onEditRegionPrice?.(row, region)}
          />
        ),
      }),
    ),
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
    <TableScrollFrame
      className={`otari-offered-models-table${hasRegions ? " otari-offered-models-table--regions" : ""}`}
    >
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
