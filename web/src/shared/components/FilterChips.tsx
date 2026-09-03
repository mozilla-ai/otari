import { Button } from "@heroui/react"
import type { ReactNode } from "react"
import { useId, useState } from "react"
import { DismissChip } from "./surface"

// Active filters shown as removable pills, with the pickers tucked behind an
// "Add filter" toggle. This is the log-tool convention (CloudWatch, Railway):
// keep the applied filters visible at all times (especially on mobile, where the
// picker row is otherwise hidden), and let a chip's ✕ clear just that filter.

export interface FilterChip {
  // Stable key for the collection.
  key: string
  // The dimension, e.g. "Model".
  label: string
  // The human-readable current value, e.g. "gpt-5.6".
  value: string
  // Accessible name for the ✕. Defaults to naming the dimension, which is enough
  // while a dimension has one chip; a multi-value filter renders one chip per
  // value and passes a name carrying the value, so the controls stay distinct.
  clearLabel?: string
  onClear: () => void
}

export function FilterChips({
  chips,
  children,
  onClearAll,
  start,
  end,
}: {
  chips: FilterChip[]
  // The picker controls (comboboxes/selects), revealed by "Add filter".
  children: ReactNode
  onClearAll?: () => void
  // Optional controls sharing the toggle's row: `start` renders before the
  // toggle (e.g. the date preset buttons), `end` is pushed to the right edge
  // (e.g. the window caption and refresh). Keeping them in this row saves a
  // line of vertical space on pages where filters sit beside the range picker.
  start?: ReactNode
  end?: ReactNode
}) {
  const [open, setOpen] = useState(false)
  const regionId = useId()

  return (
    <div className="flex flex-col gap-2">
      {/* `otari-toolbar`: this row IS the named dense place, so its controls
          take the 38px height a filter above a table sits at. It was a toolbar
          in everything but the class. */}
      <div className="otari-toolbar flex flex-wrap items-center gap-2">
        {start}
        <Button
          size="sm"
          variant="outline"
          onPress={() => setOpen((prev) => !prev)}
          aria-expanded={open}
          aria-controls={regionId}
        >
          {open ? "Done" : "Add filter"}
        </Button>
        {chips.map((chip) => (
          <DismissChip
            key={chip.key}
            label={chip.label}
            value={chip.value}
            onDismiss={chip.onClear}
            dismissLabel={chip.clearLabel ?? `Remove ${chip.label} filter`}
          />
        ))}
        {chips.length > 0 && onClearAll ? (
          <Button size="sm" variant="ghost" onPress={onClearAll}>
            Clear all
          </Button>
        ) : null}
        {end ? (
          <div className="ml-auto flex items-center gap-3">{end}</div>
        ) : null}
      </div>
      <div
        id={regionId}
        // The revealed pickers are the same place as the row that reveals
        // them, so they take the same dense height. Without the class they came
        // out 40px under a 38px row, which is two sizes for one control.
        className={
          open ? "otari-toolbar flex flex-wrap items-end gap-3" : "hidden"
        }
      >
        {children}
      </div>
    </div>
  )
}
