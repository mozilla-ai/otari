import { Spinner, Table } from "@heroui/react";
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { MouseEvent as ReactMouseEvent, PointerEvent as ReactPointerEvent, ReactNode } from "react";
import { Checkbox as AriaCheckbox } from "react-aria-components";
import type { Key, Selection, SortDescriptor } from "react-aria-components";

// The box visual, split out so it can hold optimistic state: react-aria only
// reports the new `isSelected` after the whole collection re-renders (O(rows)
// per click, tens to hundreds of ms on big pages or slow machines), which made
// the checkmark feel laggy. On pointerdown the visual flips immediately; the
// authoritative state catches up and clears the override, and a timeout clears
// it as a backstop if the press never lands (e.g. drag-away).
function SelectionBoxVisual({ isSelected, isIndeterminate, isDisabled }: {
  isSelected: boolean;
  isIndeterminate: boolean;
  isDisabled: boolean;
}) {
  const [flash, setFlash] = useState<boolean | null>(null);

  useEffect(() => {
    if (flash !== null && isSelected === flash) setFlash(null);
  }, [isSelected, flash]);
  useEffect(() => {
    if (flash === null) return;
    const timer = setTimeout(() => setFlash(null), 600);
    return () => clearTimeout(timer);
  }, [flash]);

  const showChecked = flash ?? (isSelected || isIndeterminate);
  return (
    <span
      onPointerDown={() => {
        if (!isDisabled) setFlash(!isSelected);
      }}
      className={`flex h-4 w-4 items-center justify-center rounded border transition-colors ${
        showChecked
          ? "border-[var(--otari-brand)] bg-[var(--otari-brand)] text-white"
          : "border-[var(--otari-line)] bg-[var(--otari-surface)]"
      } group-data-[focus-visible]:outline-2 group-data-[focus-visible]:outline-[var(--otari-brand)]`}
    >
      {isIndeterminate && flash === null ? (
        <svg viewBox="0 0 24 24" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth={3} aria-hidden>
          <line x1="6" x2="18" y1="12" y2="12" strokeLinecap="round" />
        </svg>
      ) : showChecked ? (
        <svg viewBox="0 0 24 24" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth={3} aria-hidden>
          <polyline points="5 12 10 17 19 7" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      ) : null}
    </span>
  );
}

// react-aria's own Checkbox drives table row/all selection through
// `slot="selection"`. HeroUI's Checkbox splits the control across subcomponents
// and does not cleanly forward the selection slot, so the selection box is a
// small styled react-aria Checkbox that matches the --otari tokens.
function SelectionCheckbox({ ariaLabel }: { ariaLabel: string }) {
  return (
    <AriaCheckbox slot="selection" aria-label={ariaLabel} className="group inline-flex items-center">
      {({ isSelected, isIndeterminate, isDisabled }) => (
        <SelectionBoxVisual isSelected={isSelected} isIndeterminate={isIndeterminate} isDisabled={isDisabled} />
      )}
    </AriaCheckbox>
  );
}

// Shared data table for the dashboard, built on HeroUI v3's compound Table
// (a thin wrapper over react-aria-components). Selection, keyboard grid
// navigation, sort, and column resizing all come from the library, so pages
// declare columns + rows and opt into the behaviors they need rather than
// hand-rolling a <table>. Replaces the previous bespoke table and its custom
// column-resize code.

export interface DataTableColumn<Row> {
  /** Stable column id; also the `SortDescriptor.column` value when sortable. */
  id: string;
  header: ReactNode;
  cell: (row: Row) => ReactNode;
  /** Right-align numeric columns (header and cells). */
  align?: "start" | "end";
  /** Marks the column react-aria announces as the row's header (usually the name). */
  isRowHeader?: boolean;
  allowsSorting?: boolean;
  /** Fixed/initial pixel width; only meaningful when the table is `resizable`. */
  width?: number;
  minWidth?: number;
}

export interface DataTableProps<Row> {
  /** Accessible name for the grid (required by react-aria). */
  ariaLabel: string;
  columns: DataTableColumn<Row>[];
  rows: Row[];
  getRowKey: (row: Row) => string;
  /** Shows a spinner in place of the body while the first page loads. */
  isLoading?: boolean;
  emptyContent?: ReactNode;
  selectionMode?: "none" | "multiple";
  selectedKeys?: Selection;
  onSelectionChange?: (keys: Selection) => void;
  /**
   * Rows that cannot be selected (e.g. enforced usage rows that bulk delete must
   * never touch). `disabledBehavior` is fixed to "selection", so a disabled row
   * still opens its detail on click; only its checkbox is inert.
   */
  disabledKeys?: Iterable<Key>;
  sortDescriptor?: SortDescriptor;
  onSortChange?: (descriptor: SortDescriptor) => void;
  /** Fired when a row body is activated (click / Enter), for detail drill-in. */
  onRowAction?: (key: string) => void;
  rowClassName?: (row: Row) => string | undefined;
  /** Enables draggable column resize handles. */
  resizable?: boolean;
  /**
   * Inline detail: when `detailKey` matches a row's key, `renderDetail(row)`
   * renders as a full-width row directly under that row (accordion style), so
   * the panel opens where the user clicked instead of below the table.
   *
   * The detail row is a portal-managed `<tr>` inserted next to the target row,
   * deliberately outside react-aria's collection: putting it in `items` made
   * every expand re-process the whole page of rows (~130 ms at 100 rows,
   * ~490 ms on a throttled CPU), which read as lag. Outside the collection it
   * costs O(1), never joins selection or keyboard navigation, and cannot fire
   * `onRowAction`. Keep `renderDetail` referentially stable like the other
   * render inputs.
   */
  detailKey?: string | null;
  renderDetail?: (row: Row) => ReactNode;
}

const SELECTION_COLUMN_WIDTH = 44;

// Whether the document's text selection is a real (non-empty) one anchored inside
// `root`. Used to tell "the operator was highlighting an id" from "the operator
// clicked the row": a plain click leaves a collapsed selection, and a selection
// made elsewhere on the page is not anchored here.
function hasTextSelectionIn(root: HTMLElement | null): boolean {
  if (!root) return false;
  const selection = document.getSelection();
  if (!selection || selection.isCollapsed || selection.toString().trim() === "") return false;
  return selection.anchorNode !== null && root.contains(selection.anchorNode);
}

// HeroUI's Table.Root is itself a card. Rather than wrap it in a second card
// (which left two mismatched radii and an inset gap), the `.otari-table` class
// owns the whole container's styling in globals.css: our surface + border +
// single 12px radius, the brand-tint header, and no column separators. The root's
// `overflow: hidden` (also in globals.css) clips the header and last row to that
// one radius, so the header meets the card corner exactly.

export function DataTable<Row extends object>({
  ariaLabel,
  columns,
  rows,
  getRowKey,
  isLoading = false,
  emptyContent = "No rows.",
  selectionMode = "none",
  selectedKeys,
  onSelectionChange,
  disabledKeys,
  sortDescriptor,
  onSortChange,
  onRowAction,
  rowClassName,
  resizable = false,
  detailKey = null,
  renderDetail,
}: DataTableProps<Row>) {
  const showSelection = selectionMode === "multiple";
  const Container = resizable ? Table.ResizableContainer : Table.ScrollContainer;
  const columnCount = columns.length + (showSelection ? 1 : 0);

  const rootRef = useRef<HTMLDivElement | null>(null);
  const [detailHost, setDetailHost] = useState<HTMLTableCellElement | null>(null);
  const detailRow = useMemo(
    () => (detailKey != null && renderDetail ? (rows.find((r) => getRowKey(r) === detailKey) ?? null) : null),
    [detailKey, renderDetail, rows, getRowKey],
  );

  // The portal host, built once per table and reused. Identity has to survive
  // the effect below re-running, which it does on every row change: a fresh
  // element each time re-points the portal, and React answers that by
  // unmounting the panel and mounting it again into the new node. That replays
  // the reveal animation and throws away whatever state the panel held. The
  // activity page hands this component a rebuilt rows array every two seconds
  // while it polls for in-flight requests, so an expanded row sat there
  // flashing and re-opening on a timer for as long as a request was running.
  const hostRef = useRef<{ row: HTMLTableRowElement; cell: HTMLTableCellElement } | null>(null);
  const ensureHost = () => {
    if (!hostRef.current) {
      const row = document.createElement("tr");
      row.className = "otari-detail-row";
      // Out of the grid semantics: without this the host is an implicit ARIA row
      // with the detail text as its name, confusing row counts and name lookups.
      // Its content stays in the accessibility tree as ordinary elements.
      row.setAttribute("role", "presentation");
      const cell = document.createElement("td");
      row.appendChild(cell);
      hostRef.current = { row, cell };
    }
    return hostRef.current;
  };

  // Host <tr> management: find the target row by its data-key and position the
  // host right after it. react-aria commits its real rows in a second render
  // pass, so the target may not exist yet when this effect first runs (e.g.
  // mounting with a detailKey already set); the MutationObserver finishes the
  // insertion as soon as the row appears. The deps re-run it (re-positioning
  // the host) whenever the row set, order, or target changes, and the host is
  // detached as soon as there is no target to sit under, so a vanished target
  // (filtered out, page flipped) leaves nothing behind.
  useLayoutEffect(() => {
    const root = rootRef.current;
    if (!root || detailKey == null || !detailRow) {
      hostRef.current?.row.remove();
      setDetailHost(null);
      return;
    }
    const { row: hostRow, cell: hostCell } = ensureHost();
    hostCell.colSpan = columnCount;

    const tryInsert = (): boolean => {
      const target = root.querySelector(`tbody tr[data-key="${CSS.escape(detailKey)}"]`);
      if (!target) return false;
      // The optimistic "opening" highlight has served its purpose once the
      // panel actually lands.
      for (const el of root.querySelectorAll(".otari-detail-opening")) el.classList.remove("otari-detail-opening");
      // Only move it when it is not already there. Re-inserting an attached
      // node detaches and re-attaches its subtree, which cancels and restarts
      // the reveal animation running inside it.
      if (target.nextSibling !== hostRow) target.after(hostRow);
      setDetailHost(hostCell);
      return true;
    };

    let observer: MutationObserver | null = null;
    if (!tryInsert()) {
      observer = new MutationObserver(() => {
        if (tryInsert()) {
          observer?.disconnect();
          observer = null;
        }
      });
      observer.observe(root, { childList: true, subtree: true });
    }
    return () => observer?.disconnect();
  }, [detailKey, detailRow, columnCount, rows, sortDescriptor]);

  // Detach on unmount. Deliberately not part of the effect above, whose cleanup
  // runs on every dependency change: removing the host there is what made a
  // re-render remount the panel.
  useEffect(() => () => hostRef.current?.row.remove(), []);

  // Row activation with instant acknowledgment: the detail panel can only land
  // after react-aria's O(rows) interaction render (~1.6 ms/row), so the clicked
  // row is highlighted in the same frame; the insert effect clears the class
  // when the panel arrives, with a timeout backstop.
  const fireRowAction = useCallback(
    (key: string) => {
      if (!onRowAction) return;
      if (renderDetail && key !== detailKey) {
        const target = rootRef.current?.querySelector(`tbody tr[data-key="${CSS.escape(key)}"]`);
        target?.classList.add("otari-detail-opening");
        setTimeout(() => target?.classList.remove("otari-detail-opening"), 1500);
      }
      onRowAction(key);
    },
    [onRowAction, renderDetail, detailKey],
  );

  // The row key for an event on an ordinary data cell, or null when the event
  // belongs to something else: checkboxes, buttons, links, inputs, and the detail
  // panel pass through untouched. Only meaningful for tables with a row action.
  const dataCellRowKey = useCallback(
    (e: { target: EventTarget | null }): string | null => {
      if (!onRowAction) return null;
      const target = e.target instanceof Element ? e.target : null;
      if (!target) return null;
      if (target.closest("label[slot=selection], button, a, input, select, textarea, .otari-detail-row")) return null;
      return target.closest("tbody tr[data-key]")?.getAttribute("data-key") ?? null;
    },
    [onRowAction],
  );

  // Where rows have a drill-in action, this component owns the pointer sequence on
  // data cells instead of react-aria's row press, for three reasons:
  //
  //   1. Its toggle selection behavior repurposes row clicks once the selection is
  //      non-empty: they extend the selection instead of firing the action
  //      (useSelectableItem's hasPrimaryAction requires an empty selection
  //      manager). For these tables the checkbox owns selection and a row click
  //      keeps opening the drill-in (the Gmail convention).
  //   2. The press toggles selection on pointer *down*, and the re-render that
  //      causes lands mid-drag and discards a nascent text selection, so no id in
  //      a row could be highlighted by hand (issue #478).
  //   3. Taking pointer down and the click together keeps the action firing
  //      exactly once: react-aria never sees a press to fire it a second time.
  //
  // Checkboxes, buttons, links, inputs, and the detail panel pass through, so
  // selection, row actions, and the panel's own controls behave normally.
  // Keyboard activation is untouched: Enter still routes through Table.Content's
  // onRowAction. Tables with no row action keep react-aria's press as-is; there,
  // only a CopyableValue (which stops the press on itself) is drag-selectable.

  // Rows render through react-aria's items-collection path so each row element
  // is cached per row object: a selection toggle re-renders only the affected
  // row instead of the whole page of rows (which made checkbox clicks lag by
  // whole seconds at large page sizes). The cache is invalidated when any input
  // that changes row rendering does (the `dependencies` on Table.Body below),
  // so callers must keep `columns`, `getRowKey`, and `rowClassName` (if used)
  // referentially stable across unrelated re-renders for the cache to pay off;
  // an inline arrow for any of them rebuilds every row on each render.
  const renderRow = useCallback(
    (row: Row) => {
      const key = getRowKey(row);
      return (
        <Table.Row key={key} id={key} className={rowClassName?.(row)}>
          {showSelection ? (
            <Table.Cell>
              <SelectionCheckbox ariaLabel="Select row" />
            </Table.Cell>
          ) : null}
          {columns.map((col) => (
            <Table.Cell key={col.id} className={col.align === "end" ? "text-right tabular-nums" : undefined}>
              {col.cell(row)}
            </Table.Cell>
          ))}
        </Table.Row>
      );
    },
    [getRowKey, rowClassName, showSelection, columns],
  );

  return (
    <Table.Root ref={rootRef} className="otari-table">
      <Container
        className="overflow-x-auto"
        onPointerDownCapture={(e: ReactPointerEvent) => {
          if (dataCellRowKey(e) != null) e.stopPropagation();
        }}
        onMouseDownCapture={(e: ReactMouseEvent) => {
          // react-aria falls back to mouse events where PointerEvent is
          // unavailable; the press (and its selection toggle) starts here.
          if (dataCellRowKey(e) != null) e.stopPropagation();
        }}
        onClickCapture={(e: ReactMouseEvent) => {
          const key = dataCellRowKey(e);
          if (key == null) return;
          // Swallowed either way, so react-aria's row press never fires a second
          // action. A click that ended a text drag inside the table is a
          // selection, not an activation: cells are selectable by design (see
          // globals.css), and drilling in mid-highlight both loses the selection
          // and moves the page under the operator, so the action is skipped for
          // that click only. Deliberately scoped to the click path: the same
          // check in fireRowAction would also swallow Enter on a focused row,
          // which is a deliberate activation even with an id still highlighted.
          e.stopPropagation();
          if (!hasTextSelectionIn(rootRef.current)) fireRowAction(key);
        }}
      >
        <Table.Content
          aria-label={ariaLabel}
          className="w-full text-sm"
          selectionMode={selectionMode}
          selectionBehavior="toggle"
          disabledBehavior="selection"
          selectedKeys={selectedKeys}
          onSelectionChange={onSelectionChange}
          disabledKeys={disabledKeys}
          sortDescriptor={sortDescriptor}
          onSortChange={onSortChange}
          onRowAction={onRowAction ? (key) => fireRowAction(String(key)) : undefined}
        >
          <Table.Header>
            {showSelection ? (
              <Table.Column width={SELECTION_COLUMN_WIDTH} minWidth={SELECTION_COLUMN_WIDTH}>
                <SelectionCheckbox ariaLabel="Select all rows" />
              </Table.Column>
            ) : null}
            {columns.map((col) => (
              <Table.Column
                key={col.id}
                id={col.id}
                isRowHeader={col.isRowHeader}
                allowsSorting={col.allowsSorting}
                width={col.width}
                minWidth={col.minWidth}
                className={col.align === "end" ? "text-right" : undefined}
              >
                {({ sortDirection }) => (
                  <div className={`flex items-center gap-1 ${col.align === "end" ? "justify-end" : ""}`}>
                    {col.allowsSorting ? (
                      <Table.SortableColumnHeader sortDirection={sortDirection}>
                        {col.header}
                      </Table.SortableColumnHeader>
                    ) : (
                      <span>{col.header}</span>
                    )}
                    {resizable ? <Table.ColumnResizer className="ml-auto cursor-col-resize px-1" /> : null}
                  </div>
                )}
              </Table.Column>
            ))}
          </Table.Header>
          <Table.Body
            items={isLoading && rows.length === 0 ? [] : rows}
            dependencies={[renderRow]}
            renderEmptyState={() => (
              <div className="px-4 py-10 text-center text-[var(--otari-muted)]">
                {isLoading ? (
                  <span className="inline-flex items-center gap-2">
                    <Spinner size="sm" /> Loading…
                  </span>
                ) : (
                  emptyContent
                )}
              </div>
            )}
          >
            {renderRow}
          </Table.Body>
        </Table.Content>
      </Container>
      {detailHost && detailRow && renderDetail
        ? createPortal(
            // Keyed by the row it belongs to. The host node is stable now, so
            // without this the panel would be reconciled across a jump from one
            // row to another and keep the previous row's state (RouterReadiness
            // seeds its user picker from its props once, at mount). A poll that
            // rebuilds `rows` leaves detailKey alone, so the fix above still
            // holds: only a different row remounts, which is also what replays
            // the reveal animation under the newly opened row.
            <div key={detailKey} className="otari-detail-reveal">
              <div>{renderDetail(detailRow)}</div>
            </div>,
            detailHost,
          )
        : null}
    </Table.Root>
  );
}
