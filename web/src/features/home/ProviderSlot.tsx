import type { SpendRow } from "@/features/spend/useSpend"
import {
  anyProviderMark,
  ProviderMark,
} from "@/shared/components/marks/BrandMark"

/**
 * Whether a table of provider rows reserves a mark slot: only when one of its
 * providers has a mark, so a table of tiles does not cost every row an indent.
 */
export function hasProviderSlot(rows: readonly SpendRow[]): boolean {
  return anyProviderMark(
    rows.flatMap((row) => (row.filterKey ? [row.filterKey] : [])),
  )
}

/**
 * A provider row's mark, or a lettermark tile when it has none. A row that is
 * no one provider (the fold, the requests that recorded none) keeps the slot
 * empty: a tile there would read as a provider of that name.
 */
export function ProviderSlot({ row }: { row: SpendRow }) {
  return row.filterKey ? (
    <ProviderMark providerId={row.filterKey} label={row.name} />
  ) : (
    <span aria-hidden="true" className="size-4 shrink-0" />
  )
}
