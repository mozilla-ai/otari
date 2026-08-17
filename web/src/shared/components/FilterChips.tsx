import { Button } from "@heroui/react"
import type { ReactNode } from "react"
import { useId, useState } from "react"

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
      <div className="flex flex-wrap items-center gap-2">
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
          <span
            key={chip.key}
            className="inline-flex items-center gap-1 rounded-full border border-[var(--otari-line)] bg-[var(--otari-brand-tint)] py-0.5 pl-2.5 pr-1 text-xs text-[var(--otari-brand-dark)]"
          >
            <span className="text-[var(--otari-muted)]">{chip.label}:</span>
            <span className="font-medium">{chip.value}</span>
            <button
              type="button"
              onClick={chip.onClear}
              aria-label={chip.clearLabel ?? `Remove ${chip.label} filter`}
              className="ml-0.5 inline-flex h-4 w-4 items-center justify-center rounded-full text-[var(--otari-muted)] outline-none hover:bg-[var(--otari-line)] hover:text-[var(--otari-ink)] focus-visible:ring-2 focus-visible:ring-[var(--otari-brand)]"
            >
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                className="h-3 w-3"
                aria-hidden="true"
              >
                <path d="M6 6l12 12M18 6L6 18" strokeLinecap="round" />
              </svg>
            </button>
          </span>
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
        className={open ? "flex flex-wrap items-end gap-3" : "hidden"}
      >
        {children}
      </div>
    </div>
  )
}
