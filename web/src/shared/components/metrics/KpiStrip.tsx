import type { ReactNode } from "react"
import { Section } from "@/shared/components/layout/Section"

/**
 * Five equal cells divided by vertical rules, between horizontal ones. Equal
 * rather than content-sized so the divisions land on a rhythm rather than
 * wherever the longest label happens to end.
 */
export function KpiStrip({
  children,
  empty,
}: {
  children: ReactNode
  empty: boolean
}) {
  return (
    <Section
      // Auto rows in groups of four, one group per row of cells, so the cells
      // below can subgrid onto them and line their four parts up with each
      // other. Without it a label that wraps makes its own cell taller and
      // drops its value below the others'.
      className="border-y border-border"
      contentClassName="grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-5"
      // The graphic row is dropped uniformly in the empty state, so the strip
      // gets shorter without any cell changing shape relative to its neighbors.
      data-empty={empty ? "true" : undefined}
    >
      {children}
    </Section>
  )
}
