import { Spinner } from "@heroui/react";
import {
  createContext,
  type PointerEvent as ReactPointerEvent,
  type KeyboardEvent as ReactKeyboardEvent,
  type ReactNode,
  useCallback,
  useContext,
  useRef,
  useState,
} from "react";

// Lightweight semantic table primitives. HeroUI v3 ships a full react-aria
// Table, but the dashboard's tables are read-mostly and don't need selection or
// keyboard grid navigation, so a styled <table> keeps the markup simple.

// Columns are resizable by dragging the handle on a header's right edge. Widths
// stay in memory (per mounted table) rather than persisted: the goal is to let an
// operator widen a cramped column while they work, not to remember a layout.
const MIN_COLUMN_WIDTH = 60;
const COLUMN_RESIZE_STEP = 24;

interface ColumnResize {
  // Per-column pixel widths once the table has switched to a fixed layout, or
  // null while it is still auto-sized (its original, content-driven behavior).
  widths: number[] | null;
  // Pin every header cell to its current rendered width, so switching to a fixed
  // layout does not reflow the columns before one is dragged. A no-op once pinned.
  pinWidths: (headerRow: HTMLTableRowElement) => void;
  setColumnWidth: (index: number, width: number) => void;
}

const ColumnResizeContext = createContext<ColumnResize | null>(null);

export function Table({ children }: { children: ReactNode }) {
  const [widths, setWidths] = useState<number[] | null>(null);

  const pinWidths = useCallback((headerRow: HTMLTableRowElement) => {
    setWidths((current) => {
      if (current) {
        return current;
      }
      return Array.from(headerRow.cells, (cell) => Math.max(MIN_COLUMN_WIDTH, Math.round(cell.getBoundingClientRect().width)));
    });
  }, []);

  const setColumnWidth = useCallback((index: number, width: number) => {
    setWidths((current) => {
      if (!current) {
        return current;
      }
      const next = current.slice();
      next[index] = Math.max(MIN_COLUMN_WIDTH, Math.round(width));
      return next;
    });
  }, []);

  const resize: ColumnResize = { widths, pinWidths, setColumnWidth };
  // Once fixed, the table sizes to the exact sum of its columns so shrinking one
  // never redistributes slack into the others; the wrapper scrolls if that
  // overflows. Before the first resize it is width-full and auto-sized, so an
  // untouched table renders exactly as before.
  const totalWidth = widths ? widths.reduce((sum, width) => sum + width, 0) : null;

  return (
    <ColumnResizeContext.Provider value={resize}>
      <div className="overflow-x-auto rounded-xl border border-[var(--otari-line)] bg-[var(--otari-surface)]">
        <table
          className={`border-collapse text-sm ${widths ? "table-fixed" : "w-full"}`}
          style={totalWidth ? { width: `${totalWidth}px` } : undefined}
        >
          {widths ? (
            <colgroup>
              {widths.map((width, index) => (
                // Columns are positional and never reorder, so the index is a stable key.
                <col key={index} style={{ width: `${width}px` }} />
              ))}
            </colgroup>
          ) : null}
          {children}
        </table>
      </div>
    </ColumnResizeContext.Provider>
  );
}

export function THead({ children }: { children: ReactNode }) {
  return (
    <thead className="border-b border-[var(--otari-line)] bg-[var(--otari-brand-tint)] text-left">
      {children}
    </thead>
  );
}

// The drag handle on a header cell's right edge. Resolves its column from the DOM
// (the parent cell's position in its row) so it needs no per-column wiring, and
// mirrors the sidebar resizer: pointer drag, plus arrow keys for keyboard users.
function ColumnResizeHandle({ resize }: { resize: ColumnResize }) {
  const ref = useRef<HTMLSpanElement>(null);
  const drag = useRef<{ index: number; startX: number; startWidth: number } | null>(null);

  const columnFromDom = (): { index: number; row: HTMLTableRowElement; width: number } | null => {
    const cell = ref.current?.closest("th");
    const row = cell?.parentElement;
    if (!cell || !(row instanceof HTMLTableRowElement)) {
      return null;
    }
    return { index: cell.cellIndex, row, width: cell.getBoundingClientRect().width };
  };

  const startResize = (event: ReactPointerEvent<HTMLSpanElement>) => {
    const column = columnFromDom();
    if (!column) {
      return;
    }
    // Keep the drag off the header's sort button and the row's click handler.
    event.preventDefault();
    event.stopPropagation();
    resize.pinWidths(column.row);
    ref.current?.setPointerCapture(event.pointerId);
    drag.current = { index: column.index, startX: event.clientX, startWidth: column.width };
  };

  const moveResize = (event: ReactPointerEvent<HTMLSpanElement>) => {
    const state = drag.current;
    if (!state || !ref.current?.hasPointerCapture(event.pointerId)) {
      return;
    }
    resize.setColumnWidth(state.index, state.startWidth + (event.clientX - state.startX));
  };

  const endResize = (event: ReactPointerEvent<HTMLSpanElement>) => {
    if (ref.current?.hasPointerCapture(event.pointerId)) {
      ref.current.releasePointerCapture(event.pointerId);
    }
    drag.current = null;
  };

  const nudgeResize = (event: ReactKeyboardEvent<HTMLSpanElement>) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") {
      return;
    }
    const column = columnFromDom();
    if (!column) {
      return;
    }
    event.preventDefault();
    resize.pinWidths(column.row);
    const base = resize.widths?.[column.index] ?? column.width;
    resize.setColumnWidth(column.index, base + (event.key === "ArrowLeft" ? -COLUMN_RESIZE_STEP : COLUMN_RESIZE_STEP));
  };

  return (
    <span
      ref={ref}
      role="separator"
      aria-orientation="vertical"
      aria-label="Resize column"
      title="Drag to resize"
      tabIndex={0}
      onPointerDown={startResize}
      onPointerMove={moveResize}
      onPointerUp={endResize}
      onKeyDown={nudgeResize}
      className="group absolute top-0 right-0 z-10 flex h-full w-2 cursor-col-resize touch-none select-none justify-end focus:outline-none"
    >
      {/* Always-visible divider so the grab point is discoverable; it brightens
          to the brand color and thickens on hover or keyboard focus. */}
      <span
        aria-hidden
        className="h-full w-px bg-[var(--otari-line)] transition-all group-hover:w-0.5 group-hover:bg-[var(--otari-brand)] group-focus-visible:w-0.5 group-focus-visible:bg-[var(--otari-brand)]"
      />
    </span>
  );
}

export function Th({
  children,
  className = "",
  ariaSort,
}: {
  children: ReactNode;
  className?: string;
  // Announces sort state to assistive tech on a sortable column header.
  ariaSort?: "ascending" | "descending" | "none";
}) {
  const resize = useContext(ColumnResizeContext);
  return (
    <th
      aria-sort={ariaSort}
      className={`relative px-4 py-2.5 font-semibold text-[var(--otari-ink)] whitespace-nowrap ${className}`}
    >
      {children}
      {resize ? <ColumnResizeHandle resize={resize} /> : null}
    </th>
  );
}

export function Td({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <td className={`px-4 py-2.5 align-middle ${className}`}>{children}</td>;
}

export function Tr({
  children,
  className = "",
  onClick,
  selected,
}: {
  children: ReactNode;
  className?: string;
  // When set, the row is clickable (used for row-selection tables).
  onClick?: () => void;
  selected?: boolean;
}) {
  return (
    <tr
      onClick={onClick}
      // A clickable row is the only path to the detail panel, so it must be
      // reachable and operable by keyboard (WCAG 2.1.1), not just the mouse.
      onKeyDown={
        onClick
          ? (event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                onClick();
              }
            }
          : undefined
      }
      tabIndex={onClick ? 0 : undefined}
      aria-selected={selected}
      className={`border-b border-[var(--otari-line)] last:border-b-0 ${
        onClick
          ? "cursor-pointer focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-[var(--otari-brand)]"
          : ""
      } ${selected ? "bg-[var(--otari-brand-tint)]" : "hover:bg-[var(--otari-bg)]"} ${className}`}
    >
      {children}
    </tr>
  );
}

export function TableMessage({ colSpan, children }: { colSpan: number; children: ReactNode }) {
  return (
    <tr>
      <td colSpan={colSpan} className="px-4 py-10 text-center text-[var(--otari-muted)]">
        {children}
      </td>
    </tr>
  );
}

export function LoadingRow({ colSpan }: { colSpan: number }) {
  return (
    <TableMessage colSpan={colSpan}>
      <span className="inline-flex items-center gap-2">
        <Spinner size="sm" /> Loading…
      </span>
    </TableMessage>
  );
}
