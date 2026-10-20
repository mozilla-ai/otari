import { useState } from "react"
import {
  PAGE_SIZE_OPTIONS,
  type TablePaginationProps,
} from "@/design-system/data/TablePagination"

/**
 * Pages rows the client already holds, for a table whose endpoint returns
 * them all at once. Spread `pagination` into `TablePagination`, pass `pageRows`
 * to the table and `pagination.page` as its `scrollResetKey`.
 *
 * `resetKey` returns to the first page when it changes. It is compared by
 * identity, so pass a primitive that changes only with the filters (a
 * serialized filter set, say), never the rows, which a refetch replaces. A
 * page past the end, after the rows shrink, becomes the last one.
 */
export function useClientPagination<Row>(rows: Row[], resetKey?: unknown) {
  const [page, setPage] = useState(0)
  const [pageSize, setPageSize] = useState(PAGE_SIZE_OPTIONS[0])
  const [seenKey, setSeenKey] = useState(resetKey)
  const lastPage = Math.max(0, Math.ceil(rows.length / pageSize) - 1)
  let shown = page
  if (resetKey !== seenKey) {
    // Ahead of the clamp below, so a filter that also shrinks the rows lands
    // on the first page rather than the new last one.
    setSeenKey(resetKey)
    setPage(0)
    shown = 0
  } else if (page > lastPage) {
    // Written back, not only shown: left in state, a page past the end would
    // return the moment the rows grew again, with no press from anyone.
    setPage(lastPage)
    shown = lastPage
  }
  const pageRows = rows.slice(shown * pageSize, (shown + 1) * pageSize)
  const pagination: Pick<
    TablePaginationProps,
    | "page"
    | "pageSize"
    | "total"
    | "rowsOnPage"
    | "onPageChange"
    | "onPageSizeChange"
  > = {
    page: shown,
    pageSize,
    total: rows.length,
    rowsOnPage: pageRows.length,
    onPageChange: setPage,
    onPageSizeChange: (size) => {
      setPageSize(size)
      setPage(0)
    },
  }
  return { pageRows, pagination }
}
