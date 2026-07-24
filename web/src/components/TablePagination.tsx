import { Button, Spinner } from "@heroui/react";
import { useEffect, useId, useState } from "react";

import { FilterSelect } from "@/components/ui";

// Shared pager for the dashboard tables: rows-per-page on the left, a truthful
// "range of total" summary in the middle, and first / prev / type-a-page / next
// / last controls on the right. Pages are 0-based in props; the type-a-page box
// shows 1-based numbers to the operator.

export const PAGE_SIZE_OPTIONS = [25, 50, 100, 250, 500];

export interface TablePaginationProps {
  /** 0-based current page. */
  page: number;
  pageSize: number;
  /**
   * Exact total row count, or null when it is not known (e.g. the count request
   * failed). When null the "last" jump is disabled and next falls back to
   * `hasNextFallback`.
   */
  total: number | null;
  /** Rows currently on screen, so the range end stays truthful without an exact total. */
  rowsOnPage: number;
  onPageChange: (page: number) => void;
  onPageSizeChange: (size: number) => void;
  pageSizeOptions?: number[];
  isFetching?: boolean;
  /** With an unknown total, whether a next page is assumed to exist (usually rowsOnPage === pageSize). */
  hasNextFallback?: boolean;
}

export function TablePagination({
  page,
  pageSize,
  total,
  rowsOnPage,
  onPageChange,
  onPageSizeChange,
  pageSizeOptions = PAGE_SIZE_OPTIONS,
  isFetching = false,
  hasNextFallback = false,
}: TablePaginationProps) {
  const sizeSelectId = useId();
  const pageCount = total != null ? Math.max(1, Math.ceil(total / pageSize)) : null;
  const isFirst = page === 0;
  const isLast = pageCount != null ? page >= pageCount - 1 : !hasNextFallback;

  const rangeStart = rowsOnPage > 0 ? page * pageSize + 1 : 0;
  const rangeEnd = page * pageSize + rowsOnPage;
  const summary =
    total != null
      ? total === 0
        ? "0 of 0"
        : `${rangeStart.toLocaleString()}–${rangeEnd.toLocaleString()} of ${total.toLocaleString()}`
      : rowsOnPage > 0
        ? `${rangeStart.toLocaleString()}–${rangeEnd.toLocaleString()}`
        : "0";

  // Local, editable page box synced to `page`; commits on Enter or blur so
  // intermediate keystrokes do not refetch on every digit.
  const [pageText, setPageText] = useState(String(page + 1));
  useEffect(() => {
    setPageText(String(page + 1));
  }, [page]);

  const commitPage = () => {
    const parsed = Number.parseInt(pageText, 10);
    if (Number.isNaN(parsed)) {
      setPageText(String(page + 1));
      return;
    }
    const upper = pageCount ?? Number.MAX_SAFE_INTEGER;
    const clamped = Math.min(Math.max(parsed, 1), upper);
    if (clamped - 1 !== page) {
      onPageChange(clamped - 1);
    } else {
      setPageText(String(page + 1));
    }
  };

  return (
    <div className="flex flex-wrap items-center justify-between gap-3">
      <div className="flex items-center gap-2">
        <label htmlFor={sizeSelectId} className="text-sm text-[var(--otari-muted)]">
          Rows
        </label>
        <FilterSelect
          id={sizeSelectId}
          ariaLabel="Rows per page"
          value={String(pageSize)}
          onChange={(value) => onPageSizeChange(Number.parseInt(value, 10))}
          options={pageSizeOptions.map((size) => ({ value: String(size), label: String(size) }))}
        />
        {isFetching ? <Spinner size="sm" /> : null}
      </div>

      <div className="flex items-center gap-3">
        <span className="text-sm text-[var(--otari-muted)] tabular-nums">{summary}</span>
        <div className="flex items-center gap-1">
          <Button size="sm" variant="outline" aria-label="First page" isDisabled={isFirst} onPress={() => onPageChange(0)}>
            «
          </Button>
          <Button
            size="sm"
            variant="outline"
            aria-label="Previous page"
            isDisabled={isFirst}
            onPress={() => onPageChange(page - 1)}
          >
            ‹
          </Button>
          <span className="inline-flex items-center gap-1 text-sm text-[var(--otari-muted)]">
            <input
              aria-label="Page number"
              inputMode="numeric"
              value={pageText}
              onChange={(event) => setPageText(event.target.value.replace(/[^0-9]/g, ""))}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.currentTarget.blur();
                }
              }}
              onBlur={commitPage}
              className="w-12 rounded-lg border border-[var(--otari-line)] bg-[var(--otari-bg)] px-2 py-1 text-center text-sm text-[var(--otari-ink)] tabular-nums focus:border-[var(--otari-brand)] focus:outline-none"
            />
            {pageCount != null ? <span className="tabular-nums">/ {pageCount.toLocaleString()}</span> : null}
          </span>
          <Button
            size="sm"
            variant="outline"
            aria-label="Next page"
            isDisabled={isLast}
            onPress={() => onPageChange(page + 1)}
          >
            ›
          </Button>
          <Button
            size="sm"
            variant="outline"
            aria-label="Last page"
            isDisabled={pageCount == null || isLast}
            onPress={() => pageCount != null && onPageChange(pageCount - 1)}
          >
            »
          </Button>
        </div>
      </div>
    </div>
  );
}
