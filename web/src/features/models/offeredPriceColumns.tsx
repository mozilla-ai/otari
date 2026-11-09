import type { DataTableColumn } from "@/design-system/data/DataTable"
import { Tooltip } from "@/design-system/overlays/Tooltip"

import { type OfferedRates, offeredPrice, type PriceSide } from "./offeredKinds"

const UNITS =
  "$ per million tokens for text models. Image, audio and video models show their own unit (per image, per second, per minute)."

/** A column header that says, on hover, what its prices are per. */
function unitHeader(label: string) {
  return (
    <Tooltip content={UNITS} delay={0}>
      <span className="underline decoration-dotted underline-offset-4">
        {label}
      </span>
    </Tooltip>
  )
}

/**
 * One price cell. A token price is shown bare. Any other price shows its
 * amount, underlined, and names what it prices and its unit on hover. A phone
 * cannot hover, so below `md` the unit is shown beside the amount. An absent
 * price is a dash, never a zero, which is a real price.
 */
function priceCell(row: OfferedRates, side: PriceSide) {
  const price = offeredPrice(row, side)
  if (price === undefined) return <span className="text-subtle">—</span>
  const amount = (
    <span className="text-mono-caption tabular-nums">{price.amount}</span>
  )
  if (price.rate === undefined) return amount
  return (
    <Tooltip
      delay={0}
      content={
        <span className="flex flex-col">
          <span>{price.title}</span>
          <span className="tabular-nums">{price.rate}</span>
        </span>
      }
    >
      <span className="text-mono-caption tabular-nums underline decoration-dotted underline-offset-4">
        {price.amount}
        <span aria-hidden className="md:hidden">
          {price.per}
        </span>
        {/* The unit, for a reader that never hovers. */}
        <span className="sr-only">{`, ${price.title}: ${price.rate}`}</span>
      </span>
    </Tooltip>
  )
}

/** The price columns of a provider's model list, the same for every kind of model. */
export function offeredPriceColumns<
  Row extends OfferedRates,
>(): DataTableColumn<Row>[] {
  return [
    {
      id: "input",
      header: unitHeader("Input"),
      align: "end",
      cell: (row) => priceCell(row, "input"),
    },
    {
      id: "output",
      header: unitHeader("Output"),
      align: "end",
      cell: (row) => priceCell(row, "output"),
    },
    {
      id: "cache_read",
      header: "Cache read",
      align: "end",
      cell: (row) => priceCell(row, "cache_read"),
    },
    {
      id: "cache_write",
      header: "Cache write",
      align: "end",
      cell: (row) => priceCell(row, "cache_write"),
    },
  ]
}
