import { Button, Spinner } from "@heroui/react"
import { useEffect, useId, useState } from "react"

import { FilterSelect, INPUT_CLASS } from "@/shared/components/ui"

// Shared pager for the dashboard tables: rows-per-page on the left, a truthful
// "range of total" summary in the middle, and first / prev / type-a-page / next
// / last controls on the right. Pages are 0-based in props; the type-a-page box
// shows 1-based numbers to the operator.

// Capped at 100: react-aria's Table re-renders every mounted row on each
// selection change, so click-to-select latency grows linearly with page size
// (~35 ms at 50 rows, ~320 ms at 500, beyond a second on a slower machine).
// Bulk actions over more rows than a page go through "select all N matching
// this filter" instead of a bigger page.
export const PAGE_SIZE_OPTIONS = [25, 50, 100]

export interface TablePaginationProps {
  /** 0-based current page. */
  page: number
  pageSize: number
  /**
   * Exact total row count, or null when it is not known (e.g. the count request
   * failed). When null the "last" jump is disabled and next falls back to
   * `hasNextFallback`.
   */
  total: number | null
  /** Rows currently on screen, so the range end stays truthful without an exact total. */
  rowsOnPage: number
  onPageChange: (page: number) => void
  onPageSizeChange: (size: number) => void
  pageSizeOptions?: number[]
  isFetching?: boolean
  /** With an unknown total, whether a next page is assumed to exist (usually rowsOnPage === pageSize). */
  hasNextFallback?: boolean
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
  const sizeSelectId = useId()
  const pageCount =
    total != null ? Math.max(1, Math.ceil(total / pageSize)) : null
  const isFirst = page === 0
  const isLast = pageCount != null ? page >= pageCount - 1 : !hasNextFallback

  const rangeStart = rowsOnPage > 0 ? page * pageSize + 1 : 0
  const rangeEnd = page * pageSize + rowsOnPage
  const summary =
    total != null
      ? total === 0
        ? "0 of 0"
        : `${rangeStart.toLocaleString()}–${rangeEnd.toLocaleString()} of ${total.toLocaleString()}`
      : rowsOnPage > 0
        ? `${rangeStart.toLocaleString()}–${rangeEnd.toLocaleString()}`
        : "0"

  // Local, editable page box synced to `page`; commits on Enter or blur so
  // intermediate keystrokes do not refetch on every digit.
  const [pageText, setPageText] = useState(String(page + 1))
  useEffect(() => {
    setPageText(String(page + 1))
  }, [page])

  const commitPage = () => {
    const parsed = Number.parseInt(pageText, 10)
    if (Number.isNaN(parsed)) {
      setPageText(String(page + 1))
      return
    }
    const upper = pageCount ?? Number.MAX_SAFE_INTEGER
    const clamped = Math.min(Math.max(parsed, 1), upper)
    if (clamped - 1 !== page) {
      onPageChange(clamped - 1)
    } else {
      setPageText(String(page + 1))
    }
  }

  return (
    <div className="otari-pagination flex flex-wrap items-center justify-between gap-3">
      <div className="flex items-center gap-2">
        <label htmlFor={sizeSelectId} className="text-sm text-muted">
          Rows
        </label>
        <FilterSelect
          id={sizeSelectId}
          ariaLabel="Rows per page"
          value={String(pageSize)}
          onChange={(value) => onPageSizeChange(Number.parseInt(value, 10))}
          options={pageSizeOptions.map((size) => ({
            value: String(size),
            label: String(size),
          }))}
        />
        {/* A background refetch is not worth announcing on every page change,
            so this stays out of the a11y tree rather than being HeroUI's own
            role="status" region. The row summary beside it is the live text. */}
        {isFetching ? <Spinner size="sm" aria-hidden="true" /> : null}
      </div>

      <div className="flex items-center gap-3">
        <span className="text-sm text-muted tabular-nums">{summary}</span>
        <div className="flex items-center gap-1">
          <Button
            size="sm"
            variant="outline"
            aria-label="First page"
            isDisabled={isFirst}
            onPress={() => onPageChange(0)}
          >
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
          <span className="inline-flex items-center gap-1 text-sm text-muted">
            <input
              aria-label="Page number"
              inputMode="numeric"
              value={pageText}
              onChange={(event) =>
                setPageText(event.target.value.replace(/[^0-9]/g, ""))
              }
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.currentTarget.blur()
                }
              }}
              onBlur={commitPage}
              // A real field, taking the pagination place's 32px rather than
              // hard-coding a height. Hand-rolling it is how it escaped the
              // 40px floor in the first place, which looked like the right
              // answer and was the right answer for the wrong reason.
              className={`w-12 text-center tabular-nums ${INPUT_CLASS}`}
            />
            {pageCount != null ? (
              <span className="tabular-nums">
                / {pageCount.toLocaleString()}
              </span>
            ) : null}
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
  )
}
