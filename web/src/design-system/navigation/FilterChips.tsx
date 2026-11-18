import { Button } from "@heroui/react"
import type { ReactNode } from "react"
import { useId, useState } from "react"
import { DismissChip } from "@/design-system/indicators/DismissChip"

// Active filters shown as removable pills, with the pickers tucked behind an
// "Add filter" toggle. This is the log-tool convention (CloudWatch):
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

/**
 * The row the filters themselves sit in. On a desk each filter is sized to
 * what it carries, so the row reads as a sentence; on a phone they take a
 * two-column grid and fill their cell, so the column keeps one right edge.
 */
const FILTER_ROW =
  "otari-toolbar flex flex-wrap items-end gap-3 max-md:grid max-md:grid-cols-2 max-md:[&>*]:w-full max-md:[&>*]:min-w-0"

export function FilterChips({
  chips,
  children,
  onClearAll,
  start,
  end,
  toggleLabel = "Add filter",
  alwaysOpen = false,
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
  // The toggle's name while closed, for a page that carries two of these and
  // needs the controls told apart.
  toggleLabel?: string
  // Pickers always shown, with no toggle: for a page whose filters are its
  // main control rather than a refinement.
  alwaysOpen?: boolean
}) {
  const [open, setOpen] = useState(false)
  const regionId = useId()

  const chipRow = (
    <>
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
    </>
  )

  if (alwaysOpen) {
    return (
      <div className="flex flex-col gap-2">
        <div className={FILTER_ROW}>
          {start}
          {children}
          {end ? (
            <div className="ml-auto flex items-center gap-3 max-md:col-span-2">
              {end}
            </div>
          ) : null}
        </div>
        {chips.length > 0 ? (
          <div className="otari-toolbar flex flex-wrap items-center gap-2 pointer-coarse:gap-y-5">
            {chipRow}
          </div>
        ) : null}
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-2">
      {/* `otari-toolbar`: this row IS the named dense place, so its controls
          take the 38px height a filter above a table sits at. It was a toolbar
          in everything but the class. */}
      <div
        className={`otari-toolbar flex flex-wrap items-center gap-2 ${
          chips.length > 0 ? "pointer-coarse:gap-y-5" : ""
        }`}
      >
        {/* On a phone `start` takes its own line, so the toggle always opens
            the next one and can take the leading pull a first item gets (see
            `.otari-toolbar` in globals.css) without overlapping anything. */}
        {start ? (
          <div className="min-w-0 max-w-full max-md:basis-full">{start}</div>
        ) : null}
        <Button
          size="sm"
          variant="ghost"
          className={start ? "max-md:-ms-[0.6875rem]" : undefined}
          onPress={() => setOpen((prev) => !prev)}
          aria-expanded={open}
          aria-controls={regionId}
        >
          {open ? "Done" : toggleLabel}
        </Button>
        {chipRow}
        {end ? (
          <div className="ml-auto flex items-center gap-3">{end}</div>
        ) : null}
      </div>
      <div
        id={regionId}
        // The revealed pickers are the same place as the row that reveals
        // them, so they take the same dense height. Without the class they came
        // out 40px under a 38px row, which is two sizes for one control.
        className={open ? FILTER_ROW : "hidden"}
      >
        {children}
      </div>
    </div>
  )
}
