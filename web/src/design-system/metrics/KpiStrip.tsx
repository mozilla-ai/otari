import type { ReactNode } from "react"
import { Section } from "@/design-system/layout/Section"

// Spelled out rather than interpolated: Tailwind scans for whole class names,
// so `xl:grid-cols-${n}` emits nothing.
const COLUMNS = {
  4: "grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-4",
  5: "grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-5",
} as const

/**
 * Equal cells divided by vertical rules, between horizontal ones. Equal rather
 * than content-sized so the divisions land on a rhythm rather than wherever the
 * longest label happens to end.
 *
 * `columns` is the number of cells the caller passes, not a layout preference:
 * a strip whose track count outruns its children leaves a blank column at the
 * end of the row.
 */
export function KpiStrip({
  children,
  empty,
  columns = 5,
}: {
  children: ReactNode
  empty: boolean
  columns?: keyof typeof COLUMNS
}) {
  return (
    <Section
      // Auto rows in groups of four, one group per row of cells, so the cells
      // below can subgrid onto them and line their four parts up with each
      // other. Without it a label that wraps makes its own cell taller and
      // drops its value below the others'.
      className="border-y border-border"
      contentClassName={COLUMNS[columns]}
      // The graphic row is dropped uniformly in the empty state, so the strip
      // gets shorter without any cell changing shape relative to its neighbors.
      data-empty={empty ? "true" : undefined}
    >
      {children}
    </Section>
  )
}
