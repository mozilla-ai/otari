import { useMemo, useState } from "react"
import type { Selection } from "react-aria-components"

import type { TraceSummary } from "@/client"
import { RefreshButton } from "@/design-system/actions/RefreshButton"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { TablePagination } from "@/design-system/data/TablePagination"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { SearchField } from "@/design-system/forms/SearchField"
import { Chip } from "@/design-system/indicators/Chip"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import { FilterChips } from "@/design-system/navigation/FilterChips"
import { FilterSelect } from "@/design-system/navigation/FilterSelect"
import { ActivityTimeline } from "@/features/activity/ActivityTimeline"
import {
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
  type RangePreset,
} from "@/shared/helpers/timeRange"
import { useUrlState } from "@/shared/helpers/urlState"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"

import { formatDuration, formatTraceTimestamp } from "./format"
import { toTrace } from "./fromApi"
import { TracePeek } from "./TracePeek"
import {
  sessionDurationMs,
  sessionName,
  shortId,
  traceKind,
} from "./traceModel"

const URL_DEFAULTS = { peek: "", peek_workspace: "" } as const
const LEVEL_OPTIONS = [
  { value: "", label: "All levels" },
  { value: "failed", label: "Errors" },
]
// The trace API's page size ceiling.
const PAGE_SIZE_OPTIONS = [25, 50, 100]
const MINUTE_MS = 60_000
// Where "All" starts its chart when nothing has been recorded yet.
const ALL_FALLBACK_MS = 30 * 24 * 3_600_000

const EMPTY = <span className="text-subtle">—</span>

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`

const COLUMNS: DataTableColumn<TraceSummary>[] = [
  {
    id: "last_activity",
    header: "Last activity",
    cell: (t) => (
      <span className="text-mono-caption whitespace-nowrap text-muted">
        {formatTraceTimestamp(t.last_activity_at)}
      </span>
    ),
  },
  {
    id: "name",
    header: "Name",
    isRowHeader: true,
    cell: (t) => (
      <span className="flex min-w-0 items-baseline gap-2">
        <span className="truncate text-body">{sessionName(t)}</span>
        <span className="text-mono-caption text-subtle">
          {shortId(t.trace_id)}
        </span>
      </span>
    ),
  },
  {
    id: "user",
    header: "User",
    cell: (t) =>
      t.user_id ? (
        <span className="block max-w-48 truncate text-mono-caption text-muted">
          {t.user_id}
        </span>
      ) : (
        EMPTY
      ),
  },
  {
    id: "levels",
    header: "Levels",
    cell: (t) =>
      t.error_count ? (
        <Chip tone="danger">{plural(t.error_count, "error")}</Chip>
      ) : (
        EMPTY
      ),
  },
  {
    id: "latency",
    header: "Latency",
    align: "end",
    cell: (t) => (
      <span className="text-mono-caption">
        {formatDuration(sessionDurationMs(t)) ?? EMPTY}
      </span>
    ),
  },
  {
    id: "tokens",
    header: "Tokens",
    align: "end",
    cell: (t) =>
      t.input_tokens || t.output_tokens ? (
        <span className="text-mono-caption whitespace-nowrap">
          {formatNumber(t.input_tokens)} → {formatNumber(t.output_tokens)} (Σ{" "}
          {formatNumber(t.input_tokens + t.output_tokens)})
        </span>
      ) : (
        EMPTY
      ),
  },
  {
    id: "cost",
    header: "Total cost",
    align: "end",
    cell: (t) =>
      t.cost ? (
        <span className="text-mono-caption">{formatCost(t.cost)}</span>
      ) : (
        EMPTY
      ),
  },
  {
    id: "tags",
    header: "Tags",
    cell: (t) => <Chip>{traceKind(t)}</Chip>,
  },
]

interface Window {
  start: number
  end: number
}

const iso = (ms: number) => new Date(ms).toISOString()

export function SessionsPage() {
  const url = useUrlState(URL_DEFAULTS)
  const { selected: workspace } = useSelectedWorkspace()
  const [query, setQuery] = useState("")
  const [level, setLevel] = useState("")
  const [extentKey, setExtentKey] = useState(ACTIVITY_DEFAULT_KEY)
  const [custom, setCustom] = useState<Window | null>(null)
  const [page, setPage] = useState(0)
  const [pageSize, setPageSize] = useState(50)
  const [selection, setSelection] = useState<Selection>(new Set())
  const [now, setNow] = useState(() => Date.now())

  const extent: RangePreset =
    findPreset(ACTIVITY_PRESETS, extentKey) ?? ACTIVITY_PRESETS[1]
  // Rounded down to the minute, so the query key holds still between renders.
  const extentStart =
    extent.seconds === null
      ? null
      : Math.floor((now - extent.seconds * 1000) / MINUTE_MS) * MINUTE_MS

  const base = {
    workspaceId: workspace?.workspace_id ?? "",
    q: query.trim(),
    failedOnly: level === "failed",
    harness: [],
  }
  // The chart covers the whole extent; a range picked on it narrows the list.
  const seriesFilters: TraceFilters = {
    ...base,
    start: extentStart === null ? "" : iso(extentStart),
  }
  const filters: TraceFilters = custom
    ? { ...base, start: iso(custom.start), end: iso(custom.end) }
    : seriesFilters

  const traces = useTraces(filters, page, pageSize)
  const count = useTraceCount(filters)
  const seriesQuery = useTraceSeries(seriesFilters, extent.bucket)
  const peekId = url.get("peek")
  const detail = useTrace(peekId, url.get("peek_workspace"))

  const points = seriesQuery.data?.points
  const chartStart =
    extentStart ??
    (points?.length
      ? Math.min(...points.map((p) => Date.parse(p.bucket)))
      : now - ALL_FALLBACK_MS)
  // The API answers only the buckets that hold a session; the chart wants every
  // bucket of the window, empty ones included.
  const series = useMemo(() => {
    const size = bucketDurationMs(extent.bucket)
    const byStart = new Map(
      (points ?? []).map((p) => [Math.floor(Date.parse(p.bucket) / size), p]),
    )
    const out = []
    for (let at = Math.floor(chartStart / size) * size; at < now; at += size) {
      const p = byStart.get(at / size)
      out.push({
        bucketStart: iso(at),
        requests: p ? p.succeeded + p.failed : 0,
        errors: p?.failed ?? 0,
      })
    }
    return out
  }, [points, chartStart, extent.bucket, now])
  const win = custom ?? { start: chartStart, end: now }

  const rows = traces.data?.items ?? []
  const openIndex = rows.findIndex((t) => t.trace_id === peekId)
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
      peek_workspace: rows.find((t) => t.trace_id === id)?.workspace_id ?? "",
    })
  const previous = openIndex > 0 ? rows[openIndex - 1].trace_id : null
  const next =
    openIndex >= 0 && openIndex < rows.length - 1
      ? rows[openIndex + 1].trace_id
      : null

  const resetPage = () => setPage(0)
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
      <PageIntro title="Sessions">
        Every agent session the gateway recorded: the turns, LLM calls and tools
        behind each one. No request or response content is stored.
      </PageIntro>

      <div className="flex flex-col gap-3">
        {traces.isError && !traces.data ? (
          <ErrorBanner error={traces.error} />
        ) : null}
        {detail.isError ? <ErrorBanner error={detail.error} /> : null}
        <ActivityTimeline
          presets={ACTIVITY_PRESETS}
          extentKey={custom ? CUSTOM_KEY : extentKey}
          onPreset={(preset) => {
            setExtentKey(preset.key)
            setCustom(null)
            resetPage()
          }}
          onSelectRange={(startIso, endIso) => {
            setCustom({ start: Date.parse(startIso), end: Date.parse(endIso) })
            resetPage()
          }}
          onSelectFull={() => {
            setCustom(null)
            resetPage()
          }}
          series={series}
          bucket={extent.bucket}
          windowStart={iso(win.start)}
          windowEnd={iso(win.end)}
          loading={seriesQuery.isPending}
          ariaLabel="Session volume over the selected window"
          unitLabel="Sessions"
        />
        <FilterChips
          chips={[]}
          start={
            <SearchField
              label="Search sessions"
              placeholder="Search by session ID"
              value={query}
              onChange={(value) => {
                setQuery(value)
                resetPage()
              }}
              className="w-72 max-w-full"
            />
          }
          end={<RefreshButton onRefresh={refresh} />}
        >
          <FilterSelect
            ariaLabel="Level"
            label="Level"
            value={level}
            onChange={(value) => {
              setLevel(value)
              resetPage()
            }}
            options={LEVEL_OPTIONS}
          />
        </FilterChips>

        <TableScrollFrame className="otari-activity-table">
          <DataTable
            ariaLabel="Sessions"
            columns={COLUMNS}
            rows={rows}
            getRowKey={(t) => t.trace_id}
            isLoading={traces.isPending && !traces.data}
            emptyContent="No sessions in this window. A request joins a session when its client names one; otherwise it is a session of its own."
            selectionMode="multiple"
            selectedKeys={selection}
            onSelectionChange={setSelection}
            onRowAction={openPeek}
            rowClassName={(t) =>
              t.trace_id === peekId ? "is-peeked" : undefined
            }
          />
        </TableScrollFrame>
        <TablePagination
          page={page}
          pageSize={pageSize}
          total={count.data?.count ?? null}
          rowsOnPage={rows.length}
          onPageChange={setPage}
          onPageSizeChange={(size) => {
            setPageSize(size)
            resetPage()
          }}
          pageSizeOptions={PAGE_SIZE_OPTIONS}
          isFetching={traces.isFetching}
          hasNextFallback={traces.data?.has_more ?? false}
          label="Sessions"
        />
      </div>

      {open ? (
        <TracePeek
          trace={open}
          onClose={() => url.patch({ peek: "", peek_workspace: "" })}
          onPrevious={previous ? () => openPeek(previous) : undefined}
          onNext={next ? () => openPeek(next) : undefined}
        />
      ) : null}
    </div>
  )
}
