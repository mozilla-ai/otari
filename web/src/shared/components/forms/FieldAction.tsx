import type { ReactNode } from "react"
import { FieldMessages } from "@/shared/components/forms/FieldMessages"

/**
 * A trailing action in a control row, aligned to the input line.
 *
 * A row of controls is laid out `items-end`, which bottom-aligns each child's
 * whole box. A field's box includes the caption line it reserves under its
 * input, so a bare button beside it lands level with the caption rather than
 * with the input it acts on. This gives the action the same trailing reserve
 * the fields have, which puts the two input lines on the same row.
 *
 * It reserves rather than nudging: no `self-*`, no padding, and nothing that
 * asks `align-items` for what it cannot do. The height is the caption role's
 * own line height, so a retune of the caption carries the action with it.
 *
 * Two limits, both real. Every field in these rows must reserve exactly one
 * caption line: a message long enough to wrap makes that one field taller and
 * drops it back out of line, and no reserve here can answer that. None of these
 * messages wraps at the widths they are used, so this is sound now rather than
 * permanently.
 *
 * And these rows are `flex-wrap`, where `items-end` aligns each flex LINE to its
 * own cross-end rather than the row as a whole. So this aligns the action within
 * whatever line it lands on, which after a wrap may hold nothing else. Correct
 * per line, not per row.
 */
export function FieldAction({ children }: { children: ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      {children}
      {/* `children` is required on FieldMessages; an empty reserve passes null
          rather than widening that signature for this one caller. */}
      <FieldMessages>{null}</FieldMessages>
    </div>
  )
}
