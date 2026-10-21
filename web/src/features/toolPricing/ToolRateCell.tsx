import { Badge } from "@/design-system/indicators/Badge"
import { formatUnitRate } from "@/shared/helpers/format"

import { type PricedRow, toPerCall } from "./toolRates"

/** A row's price per call: its own, the one it falls back to, or none. */
export function ToolRateCell({ entry }: { entry: PricedRow }) {
  const { price, fallback } = entry
  if (price.isError) return <Badge tone="warn">Could not read</Badge>
  if (price.isPending) return <span className="text-muted">…</span>
  if (price.data) {
    return (
      <span className="tabular-nums">
        {formatUnitRate(toPerCall(price.data.input_price_per_million))}
      </span>
    )
  }
  if (fallback?.data) {
    return (
      <span className="flex flex-col gap-0.5">
        <span className="tabular-nums">
          {formatUnitRate(toPerCall(fallback.data.input_price_per_million))}
        </span>
        <span className="text-caption">From {fallback.data.model_key}</span>
      </span>
    )
  }
  // Not "Unpriced" until the rate it would inherit is known to be absent too.
  if (fallback?.isError) return <Badge tone="warn">Could not read</Badge>
  if (fallback?.isPending) return <span className="text-muted">…</span>
  return <Badge tone="warn">Unpriced</Badge>
}
