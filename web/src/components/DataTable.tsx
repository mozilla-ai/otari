import { Spinner, Table } from "@heroui/react";
import { useCallback } from "react";
import type { ReactNode } from "react";
import { Checkbox as AriaCheckbox } from "react-aria-components";
import type { Key, Selection, SortDescriptor } from "react-aria-components";

// react-aria's own Checkbox drives table row/all selection through
// `slot="selection"`. HeroUI's Checkbox splits the control across subcomponents
// and does not cleanly forward the selection slot, so the selection box is a
// small styled react-aria Checkbox that matches the --otari tokens.
function SelectionCheckbox({ ariaLabel }: { ariaLabel: string }) {
  return (
    <AriaCheckbox slot="selection" aria-label={ariaLabel} className="group inline-flex items-center">
      {({ isSelected, isIndeterminate }) => (
        <span
          className={`flex h-4 w-4 items-center justify-center rounded border transition-colors ${
            isSelected || isIndeterminate
              ? "border-[var(--otari-brand)] bg-[var(--otari-brand)] text-white"
              : "border-[var(--otari-line)] bg-[var(--otari-surface)]"
          } group-data-[focus-visible]:outline-2 group-data-[focus-visible]:outline-[var(--otari-brand)]`}
        >
          {isIndeterminate ? (
            <svg viewBox="0 0 24 24" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth={3} aria-hidden>
              <line x1="6" x2="18" y1="12" y2="12" strokeLinecap="round" />
            </svg>
          ) : isSelected ? (
            <svg viewBox="0 0 24 24" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth={3} aria-hidden>
              <polyline points="5 12 10 17 19 7" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          ) : null}
        </span>
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
}: DataTableProps<Row>) {
  const showSelection = selectionMode === "multiple";
  const Container = resizable ? Table.ResizableContainer : Table.ScrollContainer;

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
    <Table.Root className="otari-table">
      <Container className="overflow-x-auto">
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
          onRowAction={onRowAction ? (key) => onRowAction(String(key)) : undefined}
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
    </Table.Root>
  );
}
