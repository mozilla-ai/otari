import type { ReactNode } from "react"
import { FiChevronLeft } from "react-icons/fi"

import { Button } from "../actions/Button"
import { EmptyMessage } from "../feedback/EmptyMessage"

/**
 * A list of records beside the one that is open: the page shape for a set of
 * things where reading one is most of the work.
 *
 * The ratio is 1:1.75 and lives here rather than at a call site, because it is
 * the whole reason the frame exists. The left column holds a name and a line
 * about it; the right one holds every fact a table would have spread across
 * lanes, plus the controls that act on the record. Two columns at equal width
 * give the reading half a page and the naming half more than it needs.
 *
 * **Below `md` one column shows at a time**, chosen by `isDetailShown`. Side by
 * side at 390px gives neither column a readable measure, and stacking them puts
 * every record above the one being read, so an operator scrolls past the list
 * to reach what they opened. The swap is a prop rather than state in here, so
 * the page's own selection stays the single source of what is open.
 *
 * Presentational: which record is selected is the caller's, and so is every
 * row. Pair with `ListDetailRow`, which is the row this frame's list is made
 * of, and whose separators the frame supplies.
 */
export function ListDetail({
  listLabel,
  listAction,
  list,
  empty,
  onEmptyPress,
  detail,
  detailLabel = "Details",
  isDetailShown,
  onShowList,
  backLabel = "Back to the list",
}: {
  /** What the left column holds, as its heading and as its region's name. */
  listLabel: string
  /** The control that adds a record, at the top of the column it adds to. */
  listAction?: ReactNode
  /** The rows, which are `ListDetailRow`s. */
  list: ReactNode
  /**
   * What stands in for the rows, and so how the frame is told there are none to
   * show: an empty list's message, or a line while the list is still loading. A
   * node cannot be counted from in here, which is why this rather than a flag.
   */
  empty?: ReactNode
  /**
   * Makes the empty column one press target, for a list whose first record is
   * the only thing to do with it. Omitted where the caller cannot write, so a
   * reader who would be refused is told rather than invited.
   */
  onEmptyPress?: () => void
  /** The open record, or what stands in for it while none is. */
  detail: ReactNode
  /**
   * Names the detail column's region, which holds whatever the caller puts in
   * it and so has no heading of its own to be named by.
   */
  detailLabel?: string
  /** Below `md`: whether the detail column is the one on screen. */
  isDetailShown: boolean
  /** Below `md`: the way back, since only one column is reachable at a time. */
  onShowList: () => void
  /** Names the list on that control, where "the list" is not what to call it. */
  backLabel?: string
}) {
  const rows =
    empty === undefined ? (
      // The frame divides its children, so a row draws no rule of its own and
      // the last one leaves none hanging over the space below it.
      <div className="flex flex-1 flex-col divide-y divide-border-subtle">
        {list}
      </div>
    ) : onEmptyPress === undefined ? (
      <EmptyMessage>{empty}</EmptyMessage>
    ) : (
      <button
        type="button"
        onClick={onEmptyPress}
        className="flex flex-1 flex-col justify-center transition-colors hover:bg-surface-alt motion-reduce:transition-none"
      >
        <EmptyMessage>{empty}</EmptyMessage>
      </button>
    )

  return (
    <div className="grid border border-border md:grid-cols-[1fr_1.75fr]">
      {/* Both halves are regions, so a reader can move between them rather
          than through one to reach the other. */}
      <section
        aria-label={listLabel}
        className={`min-w-0 flex-col ${isDetailShown ? "hidden md:flex" : "flex"}`}
      >
        <div className="flex items-center justify-between gap-3 border-b border-border px-4 py-2">
          <h2 className="text-overline">{listLabel}</h2>
          {listAction}
        </div>
        {rows}
      </section>
      <section
        aria-label={detailLabel}
        className={`min-w-0 flex-col border-border md:border-l ${
          isDetailShown ? "flex" : "hidden md:flex"
        }`}
      >
        {isDetailShown ? (
          // Only where a column is hidden. From `md` up the list is right
          // there, and a Back control beside it would point at what it is
          // sitting next to.
          <div className="border-b border-border px-2 py-1 md:hidden">
            <Button className="min-h-11" onPress={onShowList}>
              <FiChevronLeft aria-hidden="true" className="size-4" />
              {backLabel}
            </Button>
          </div>
        ) : null}
        <div className="flex min-w-0 flex-1 flex-col">{detail}</div>
      </section>
    </div>
  )
}

/**
 * One record in a `ListDetail`'s list: its name, a line about it, and the
 * actions that belong to the list rather than to the record.
 *
 * The row is not itself the press target. Its label is, and the actions sit
 * beside it, because a button inside a button is neither valid nor operable.
 *
 * **The action lane is there whether or not a row has actions**, so the
 * trailing controls form a vertical lane down the list instead of landing
 * wherever their row's label ends. Its width is the touch floor, which is also
 * the width of the one control it is sized for.
 *
 * `aria-current` rather than `aria-pressed`: this is the record being shown out
 * of a set, not a control that stays down. The selected row wears the accent as
 * a tint, which is the accent spent on selection the way an active nav row
 * spends it, and keeps the page's ink: the accent's own ink is under AA on its
 * tint.
 */
export function ListDetailRow({
  label,
  isSelected,
  onSelect,
  actions,
  children,
}: {
  /** The record's name, in the lane every row shares. */
  label: ReactNode
  isSelected: boolean
  onSelect: () => void
  /**
   * Controls acting on this record from the list, always visible: a touch
   * device has no hover, so a control revealed by one does not exist there.
   */
  actions?: ReactNode
  /** A second line under the label: what this record is, in a few words. */
  children?: ReactNode
}) {
  return (
    <div
      className={`flex items-stretch ${isSelected ? "bg-primary-subtle" : ""}`}
    >
      <button
        type="button"
        aria-current={isSelected ? "true" : undefined}
        onClick={onSelect}
        className={`flex min-h-11 min-w-0 flex-1 flex-col justify-center gap-0.5 px-4 py-2 text-left transition-colors motion-reduce:transition-none ${
          isSelected ? "" : "hover:bg-surface-alt"
        }`}
      >
        <span className="truncate text-body">{label}</span>
        {children === undefined ? null : (
          <span className="truncate text-caption">{children}</span>
        )}
      </button>
      <div className="flex min-w-11 shrink-0 items-center justify-end pr-2">
        {actions}
      </div>
    </div>
  )
}
