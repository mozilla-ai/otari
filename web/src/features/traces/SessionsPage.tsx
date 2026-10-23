import { useMemo, useState } from "react"
import type { SortDescriptor } from "react-aria-components"

import type { TraceSortKey, TraceSummary } from "@/client"
import { RefreshButton } from "@/design-system/actions/RefreshButton"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import {
  PAGE_SIZE_OPTIONS,
  TablePagination,
} from "@/design-system/data/TablePagination"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { SearchField } from "@/design-system/forms/SearchField"
import { useIsMobile } from "@/design-system/hooks/useIsMobile"
import { Chip } from "@/design-system/indicators/Chip"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import { ActivityTimeline } from "@/features/activity/ActivityTimeline"
import {
  DEFAULT_TRACE_SORT,
  type TraceFilters,
  useTrace,
  useTraceCount,
  useTraceSeries,
  useTraces,
} from "@/shared/api/traces"
import { formatCost, formatNumber } from "@/shared/helpers/format"
import {
  ACTIVITY_DEFAULT_KEY,
  ACTIVITY_PRESETS,
  bucketDurationMs,
  CUSTOM_KEY,
  findPreset,
} from "@/shared/helpers/timeRange"
import { useUrlState } from "@/shared/helpers/urlState"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"
import { useDebounced } from "@/shared/hooks/useDebounced"

import { formatDuration, formatTraceTimestamp } from "./format"
import { toTrace } from "./fromApi"
import { TracePeek } from "./TracePeek"
import {
  sessionDurationMs,
  sessionName,
  shortId,
  traceKind,
} from "./traceModel"

const DEFAULT_PAGE_SIZE = 50

// The list's filters, window and page live in the URL beside the open session
// and its picked span, so a shared link or Back returns to the list the panel
// was opened from, and a link from Activity opens on one LLM call.
const URL_DEFAULTS = {
  range: ACTIVITY_DEFAULT_KEY,
  start_date: "",
  end_date: "",
  q: "",
  page: "0",
  size: String(DEFAULT_PAGE_SIZE),
  peek: "",
  peek_workspace: "",
  span: "",
  sort: DEFAULT_TRACE_SORT.key,
  order: DEFAULT_TRACE_SORT.direction,
} as const
// The panel holds the picked span itself, so picking one need not re-render
// the list, its chart and its pager.
const UNTRACKED = ["span"] as const
// The columns the server can order by: each sort key and the column it heads.
const SORT_COLUMNS: Record<TraceSortKey, string> = {
  last_activity: "last_activity",
  errors: "levels",
  latency: "latency",
  tokens: "tokens",
  cost: "cost",
}
const isSortKey = (value: string): value is TraceSortKey =>
  Object.hasOwn(SORT_COLUMNS, value)
const sortKeyOf = (column: string) =>
  (Object.keys(SORT_COLUMNS) as TraceSortKey[]).find(
    (key) => SORT_COLUMNS[key] === column,
  )
// The API refuses a longer `q`; a pasted id past it would leave the list loading.
const SEARCH_MAX_LENGTH = 128
const MINUTE_MS = 60_000
// Where "All" starts its chart when nothing has been recorded yet.
const ALL_FALLBACK_MS = 30 * 24 * 3_600_000

const EMPTY = <span className="text-subtle">—</span>

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`

function columns(peekId: string): DataTableColumn<TraceSummary>[] {
  return [
    {
      id: "last_activity",
      allowsSorting: true,
      header: "Last activity",
      cell: (session) => (
        <span className="text-mono-caption whitespace-nowrap text-muted">
          {formatTraceTimestamp(session.last_activity_at)}
        </span>
      ),
    },
    {
      id: "name",
      header: "Name",
      isRowHeader: true,
      cell: (session) => (
        <span className="flex min-w-0 items-baseline gap-2">
          <span className="truncate text-body">{sessionName(session)}</span>
          <span className="shrink-0 text-mono-caption whitespace-nowrap text-subtle">
            {shortId(session.trace_id)}
          </span>
          {session.trace_id === peekId ? (
            <span className="sr-only">, open</span>
          ) : null}
        </span>
      ),
    },
    {
      id: "user",
      header: "User",
      cell: (session) =>
        session.user_id ? (
          <span className="block max-w-48 truncate text-mono-caption text-muted">
            {session.user_id}
          </span>
        ) : (
          EMPTY
        ),
    },
    {
      id: "levels",
      allowsSorting: true,
      header: "Levels",
      cell: (session) =>
        session.error_count ? (
          <Chip tone="danger">{plural(session.error_count, "error")}</Chip>
        ) : (
          EMPTY
        ),
    },
    {
      id: "latency",
      allowsSorting: true,
      header: "Latency",
      align: "end",
      cell: (session) => (
        <span className="text-mono-caption">
          {formatDuration(sessionDurationMs(session))}
        </span>
      ),
    },
    {
      id: "tokens",
      allowsSorting: true,
      header: "Tokens",
      align: "end",
      cell: (session) =>
        session.input_tokens || session.output_tokens ? (
          <span className="text-mono-caption whitespace-nowrap">
            {formatNumber(session.input_tokens)} →{" "}
            {formatNumber(session.output_tokens)} (Σ{" "}
            {formatNumber(session.input_tokens + session.output_tokens)})
          </span>
        ) : (
          EMPTY
        ),
    },
    {
      id: "cost",
      allowsSorting: true,
      header: "Total cost",
      align: "end",
      cell: (session) =>
        session.cost ? (
          <span className="text-mono-caption">{formatCost(session.cost)}</span>
        ) : (
          EMPTY
        ),
    },
    {
      id: "tags",
      header: "Tags",
      cell: (session) => <Chip>{traceKind(session)}</Chip>,
    },
  ]
}

interface TimeWindow {
  start: number
  end: number
}

const iso = (ms: number) => new Date(ms).toISOString()

export function SessionsPage() {
  const url = useUrlState(URL_DEFAULTS, { untracked: UNTRACKED })
  const { selected: workspace } = useSelectedWorkspace()
  const isMobile = useIsMobile()
  const [now, setNow] = useState(() => Date.now())

  const query = url.get("q")
  const page = Math.max(0, url.getNumber("page"))
  // Snapped to an offered size, as Activity does, so a hand-edited URL cannot
  // reach the API as a limit it refuses.
  const rawPageSize = url.getNumber("size")
  const pageSize = PAGE_SIZE_OPTIONS.reduce((best, option) =>
    Math.abs(option - rawPageSize) < Math.abs(best - rawPageSize)
      ? option
      : best,
  )
  const startParam = url.get("start_date")
  const endParam = url.get("end_date")
  const custom: TimeWindow | undefined =
    startParam && endParam
      ? { start: Date.parse(startParam), end: Date.parse(endParam) }
      : undefined

  const extent =
    findPreset(ACTIVITY_PRESETS, url.get("range")) ??
    findPreset(ACTIVITY_PRESETS, ACTIVITY_DEFAULT_KEY)
  const bucket = extent?.bucket ?? "hour"
  const extentSeconds = extent?.seconds ?? null
  // Rounded down to the minute, so the query key holds still between renders.
  const extentStart =
    extentSeconds === null
      ? null
      : Math.floor((now - extentSeconds * 1000) / MINUTE_MS) * MINUTE_MS

  const search = useDebounced(query.trim().slice(0, SEARCH_MAX_LENGTH))
  const base = {
    workspaceId: workspace?.workspace_id ?? "",
    q: search,
  }
  // The chart covers the whole extent; a range picked on it narrows the list.
  const seriesFilters: TraceFilters = {
    ...base,
    start: extentStart === null ? "" : iso(extentStart),
  }
  const filters: TraceFilters = custom
    ? { ...base, start: iso(custom.start), end: iso(custom.end) }
    : seriesFilters

  // A hand-edited URL naming a column the server cannot sort falls back to
  // the default rather than reaching the API as a value it refuses.
  const sortParam = url.get("sort")
  const sortKey = isSortKey(sortParam) ? sortParam : DEFAULT_TRACE_SORT.key
  const sortDirection = url.get("order") === "asc" ? "asc" : "desc"
  const sort = useMemo(
    () => ({ key: sortKey, direction: sortDirection }) as const,
    [sortKey, sortDirection],
  )
  const sortDescriptor: SortDescriptor = {
    column: SORT_COLUMNS[sortKey],
    direction: sortDirection === "asc" ? "ascending" : "descending",
  }
  // Every sortable column is a time or an amount, so a column's first press
  // puts the largest first; pressing it again flips it.
  const onSortChange = (descriptor: SortDescriptor) => {
    const key = sortKeyOf(String(descriptor.column))
    if (!key) return
    url.patch({
      sort: key,
      order:
        key === sortKey && descriptor.direction === "ascending"
          ? "asc"
          : "desc",
      page: 0,
    })
  }

  const traces = useTraces(filters, page, pageSize, sort)
  const count = useTraceCount(filters)
  const seriesQuery = useTraceSeries(seriesFilters, bucket)
  const peekId = url.get("peek")
  const detail = useTrace(peekId, url.get("peek_workspace"))

  const points = seriesQuery.data?.points
  const chartStart =
    extentStart ??
    (points?.length
      ? Math.min(...points.map((point) => Date.parse(point.bucket)))
      : now - ALL_FALLBACK_MS)
  // The API answers only the buckets that hold a session; the chart wants every
  // bucket of the window, empty ones included.
  const series = useMemo(() => {
    const size = bucketDurationMs(bucket)
    const byStart = new Map(
      (points ?? []).map((point) => [
        Math.floor(Date.parse(point.bucket) / size),
        point,
      ]),
    )
    const out = []
    for (let at = Math.floor(chartStart / size) * size; at < now; at += size) {
      const point = byStart.get(at / size)
      out.push({
        bucketStart: iso(at),
        requests: point ? point.succeeded + point.failed : 0,
        errors: point?.failed ?? 0,
      })
    }
    return out
  }, [points, chartStart, bucket, now])
  const win = custom ?? { start: chartStart, end: now }
  const tableColumns = useMemo(() => columns(peekId), [peekId])

  const rows = traces.data?.items ?? []
  const openIndex = rows.findIndex((session) => session.trace_id === peekId)
  // The row on screen, or the read's own summary for a link off this page.
  const openSummary = rows[openIndex] ?? detail.data?.summary
  const open = useMemo(
    () => (openSummary ? toTrace(openSummary, detail.data) : null),
    [openSummary, detail.data],
  )
  // A trace is keyed by its workspace and its id, so the URL keeps both.
  const openPeek = (id: string) =>
    url.patch({
      peek: id,
      peek_workspace:
        rows.find((session) => session.trace_id === id)?.workspace_id ?? "",
      span: "",
    })
  const previous = openIndex > 0 ? rows[openIndex - 1].trace_id : null
  const next =
    openIndex >= 0 && openIndex < rows.length - 1
      ? rows[openIndex + 1].trace_id
      : null

  // A minute later is a new window; inside the same minute the key holds, so
  // the reads are refetched as well.
  const refresh = () => {
    setNow(Date.now())
    void traces.refetch()
    void count.refetch()
    void seriesQuery.refetch()
  }

  return (
    <div className="flex flex-col">
      {/* Below md the panel covers the page, which then takes no focus or
          presses until it closes. */}
      <div className="flex flex-col" inert={open !== null && isMobile}>
        <PageIntro title="Sessions">
          Every agent session the gateway recorded: the turns, LLM calls and
          tools behind each one. No request or response content is stored.
        </PageIntro>

        <div className="flex flex-col gap-3">
          {traces.isError && !traces.data ? (
            <ErrorBanner error={traces.error} />
          ) : null}
          {/* With no row to open on, the panel cannot say it either. */}
          {detail.isError && !open ? (
            <ErrorBanner error={detail.error} />
          ) : null}
          <ActivityTimeline
            presets={ACTIVITY_PRESETS}
            extentKey={
              custom ? CUSTOM_KEY : (extent?.key ?? ACTIVITY_DEFAULT_KEY)
            }
            onPreset={(preset) =>
              url.patch({
                range: preset.key,
                start_date: "",
                end_date: "",
                page: 0,
              })
            }
            onSelectRange={(startIso, endIso) =>
              url.patch({ start_date: startIso, end_date: endIso, page: 0 })
            }
            onSelectFull={() =>
              url.patch({ start_date: "", end_date: "", page: 0 })
            }
            series={series}
            bucket={bucket}
            windowStart={iso(win.start)}
            windowEnd={iso(win.end)}
            loading={seriesQuery.isPending}
            ariaLabel="Session volume over the selected window"
            unitLabel="Sessions"
          />
          <div className="otari-toolbar flex flex-wrap items-center gap-2">
            <SearchField
              label="Search sessions"
              placeholder="Search by ID, name, agent or user"
              value={query}
              onChange={(value) => url.patch({ q: value, page: 0 })}
              maxLength={SEARCH_MAX_LENGTH}
              className="w-72 max-w-full"
            />
            <div className="ml-auto flex items-center gap-3">
              <RefreshButton onRefresh={refresh} />
            </div>
          </div>

          <TableScrollFrame className="otari-sessions-table">
            <DataTable
              ariaLabel="Sessions"
              columns={tableColumns}
              rows={rows}
              getRowKey={(session) => session.trace_id}
              sortDescriptor={sortDescriptor}
              onSortChange={onSortChange}
              isLoading={traces.isPending && !traces.data}
              emptyContent={
                search
                  ? `No sessions in this window match “${search}”.`
                  : "No sessions in this window. A request joins a session when its client names one; otherwise it is a session of its own."
              }
              mobileLayout="cards"
              onRowAction={openPeek}
              rowClassName={(session) =>
                session.trace_id === peekId ? "is-peeked" : undefined
              }
            />
          </TableScrollFrame>
          <TablePagination
            page={page}
            pageSize={pageSize}
            total={count.data?.count ?? null}
            rowsOnPage={rows.length}
            onPageChange={(nextPage) => url.patch({ page: nextPage })}
            onPageSizeChange={(size) => url.patch({ size, page: 0 })}
            pageSizeOptions={PAGE_SIZE_OPTIONS}
            isFetching={traces.isFetching}
            hasNextFallback={traces.data?.has_more ?? false}
            label="Sessions"
          />
        </div>
      </div>

      {open ? (
        <TracePeek
          trace={open}
          error={detail.isError ? detail.error : undefined}
          initialSpanId={url.get("span")}
          onSpanChange={(spanId) => url.patch({ span: spanId })}
          onClose={() => url.patch({ peek: "", peek_workspace: "", span: "" })}
          onPrevious={previous ? () => openPeek(previous) : undefined}
          onNext={next ? () => openPeek(next) : undefined}
        />
      ) : null}
    </div>
  )
}
