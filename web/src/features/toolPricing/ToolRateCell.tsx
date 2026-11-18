import { Badge } from "@/design-system/indicators/Badge"
import { formatUnitRate } from "@/shared/helpers/format"

import { type PricedRow, toPerCall } from "./toolRates"

/** A row's price per call, or that it has none. */
export function ToolRateCell({ entry }: { entry: PricedRow }) {
  const { price } = entry
  if (price.isError) return <Badge tone="warn">Could not read</Badge>
  if (price.isPending) return <span className="text-muted">…</span>
  if (price.data) {
    return (
      <span className="tabular-nums">
        {formatUnitRate(toPerCall(price.data.input_price_per_million))}
      </span>
    )
  }
  return <Badge tone="warn">Unpriced</Badge>
}
