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

  // Host <tr> management: find the target row by its data-key and insert the
  // host right after it. react-aria commits its real rows in a second render
  // pass, so the target may not exist yet when this effect first runs (e.g.
  // mounting with a detailKey already set); the MutationObserver finishes the
  // insertion as soon as the row appears. The deps re-run it (re-inserting at
  // the right spot) whenever the row set, order, or target changes, and
  // cleanup always removes the host, so a vanished target (filtered out, page
  // flipped) leaves nothing behind.
  useLayoutEffect(() => {
    setDetailHost(null);
    const root = rootRef.current;
    if (!root || detailKey == null || !detailRow) return;
    const hostRow = document.createElement("tr");
    hostRow.className = "otari-detail-row";
    // Out of the grid semantics: without this the host is an implicit ARIA row
    // with the detail text as its name, confusing row counts and name lookups.
    // Its content stays in the accessibility tree as ordinary elements.
    hostRow.setAttribute("role", "presentation");
    const hostCell = document.createElement("td");
    hostCell.colSpan = columnCount;
    hostRow.appendChild(hostCell);

    const tryInsert = (): boolean => {
      const target = root.querySelector(`tbody tr[data-key="${CSS.escape(detailKey)}"]`);
      if (!target) return false;
      // The optimistic "opening" highlight has served its purpose once the
      // panel actually lands.
      for (const el of root.querySelectorAll(".otari-detail-opening")) el.classList.remove("otari-detail-opening");
      target.after(hostRow);
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
    return () => {
      observer?.disconnect();
      hostRow.remove();
    };
  }, [detailKey, detailRow, columnCount, rows, sortDescriptor]);

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

  // react-aria's toggle selection behavior repurposes row clicks once the
  // selection is non-empty: they extend the selection instead of firing the
  // row action (useSelectableItem's hasPrimaryAction requires an empty
  // selection manager). For these tables the checkbox owns selection and a row
  // click must keep opening the drill-in (the Gmail convention), so while a
  // selection exists, clicks on ordinary data cells are intercepted before the
  // row's press handler sees them and routed to the row action instead.
  // Checkboxes, buttons, links, inputs, and the detail panel pass through.
  const interceptedRowKey = useCallback(
    (e: { target: EventTarget | null }): string | null => {
      if (!onRowAction) return null;
      const hasSelection = selectedKeys === "all" || (selectedKeys instanceof Set && selectedKeys.size > 0);
      if (!hasSelection) return null;
      const target = e.target instanceof Element ? e.target : null;
      if (!target) return null;
      if (target.closest("label[slot=selection], button, a, input, select, textarea, .otari-detail-row")) return null;
      return target.closest("tbody tr[data-key]")?.getAttribute("data-key") ?? null;
    },
    [onRowAction, selectedKeys],
  );

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
          if (interceptedRowKey(e) != null) e.stopPropagation();
        }}
        onMouseDownCapture={(e: ReactMouseEvent) => {
          // react-aria falls back to mouse events where PointerEvent is
          // unavailable; the press (and its selection toggle) starts here.
          if (interceptedRowKey(e) != null) e.stopPropagation();
        }}
        onClickCapture={(e: ReactMouseEvent) => {
          const key = interceptedRowKey(e);
          if (key != null) {
            e.stopPropagation();
            fireRowAction(key);
          }
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
            <div className="otari-detail-reveal">
              <div>{renderDetail(detailRow)}</div>
            </div>,
            detailHost,
          )
        : null}
    </Table.Root>
  );
}
