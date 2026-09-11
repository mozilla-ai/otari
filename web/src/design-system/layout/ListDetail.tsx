import { type ReactNode, useEffect, useRef } from "react"
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
 * the page's own selection stays the single source of what is open. Focus
 * travels with it: the column going out of view is `display: none` while it may
 * still hold focus, which drops it to `<body>` and throws a keyboard or screen
 * reader user to the top of the document at the moment they open a record.
 *
 * Presentational: which record is selected is the caller's, and so is every
 * row. Pair with `ListDetailRow`, which is the row this frame's list is made
 * of, and whose separators the frame supplies.
 */
export function ListDetail({
  listLabel,
  listAction,
  list,
  isEmpty = false,
  empty,
  emptyAction,
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
  /**
   * The rows. A node rather than data plus a row renderer, so a caller keeps
   * its own record type and can put something that is not a row between two.
   */
  list: ReactNode
  /**
   * Whether the list has nothing to show. Separate from `empty` because a node
   * is not a count: as one prop, `empty={cond ? "None yet." : null}` read as
   * "empty" and blanked a list that had rows in it.
   */
  isEmpty?: boolean
  /** What the column says while `isEmpty`: no records, or none loaded yet. */
  empty?: ReactNode
  /**
   * The control offered beside that message, for a list whose first record is
   * the only thing to do with it. A slot like `listAction`, so it is a named
   * button rather than a column-sized press target: the column is not a control
   * and a button the height of one announces nothing.
   */
  emptyAction?: ReactNode
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
  const backRef = useRef<HTMLButtonElement>(null)
  const listRef = useRef<HTMLElement>(null)
  const wasDetailShown = useRef(isDetailShown)

  useEffect(() => {
    // Only on a change, so arriving on the page does not take focus off
    // whatever the operator was doing.
    if (wasDetailShown.current === isDetailShown) return
    wasDetailShown.current = isDetailShown
    // No breakpoint is read here, and deliberately: the control moved to is
    // itself hidden from `md` up, so `focus()` is a no-op there and CSS decides.
    // A copy of `AppShell`'s query would be a second literal of it, which is
    // the kind that goes stale on one side only.
    const target = isDetailShown
      ? backRef.current
      : (listRef.current?.querySelector<HTMLElement>('[aria-current="true"]') ??
        listRef.current)
    target?.focus()
  }, [isDetailShown])

  const rows = isEmpty ? (
    <EmptyMessage>
      <span className="flex flex-col items-center gap-3">
        {empty}
        {emptyAction}
      </span>
    </EmptyMessage>
  ) : (
    // The frame divides its children, so a row draws no rule of its own and
    // the last one leaves none hanging over the space below it.
    <div className="flex flex-1 flex-col divide-y divide-border-subtle">
      {list}
    </div>
  )

  return (
    <div className="grid border border-border md:grid-cols-[1fr_1.75fr]">
      {/* Both halves are regions, so a reader can move between them rather
          than through one to reach the other. */}
      <section
        ref={listRef}
        aria-label={listLabel}
        // Focusable only programmatically, as the landing place when the column
        // comes back with no record current. Focus has to land somewhere in the
        // column that just appeared; `<body>` is the failure this exists to
        // avoid, not an acceptable fallback.
        tabIndex={-1}
        className={`min-w-0 flex-col outline-none ${isDetailShown ? "hidden md:flex" : "flex"}`}
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
            <Button ref={backRef} className="min-h-11" onPress={onShowList}>
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
 * wherever their row's label ends. Fixed at the touch floor rather than
 * floored at it: as `min-w-11` a 44px control plus the lane's own padding came
 * to 52px, so a row with an action and a row without ended their labels 8px
 * apart, which is the one thing a lane exists to prevent. Whatever inset the
 * control wants is the control's own padding.
 *
 * `aria-current` rather than `aria-pressed`: this is the record being shown out
 * of a set, not a control that stays down.
 *
 * The selected row is the accent tint with the page's own ink, which is what
 * the selected row of the models table wears (`features/models/ModelsPage`).
 * Not the active nav row's treatment, which is a neutral fill and an ink edge
 * and is documented in `app/nav/rowStyles` as deliberately not a tint. And not
 * the accent's own ink on top: `--color-primary` is under AA on
 * `--color-primary-subtle`, and the darker step that clears it,
 * `--color-primary-subtle-foreground`, is for the small text of a chip or a nav
 * item rather than for a row's label.
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
      <div className="flex w-11 shrink-0 items-center justify-center">
        {actions}
      </div>
    </div>
  )
}
