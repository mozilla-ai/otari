import { useMemo, useState } from "react"

import type { TraceSummary } from "@/client"
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

const DEFAULT_PAGE_SIZE = 50

// The list's filters, window and page live in the URL beside the open session,
// so a shared link or Back returns to the list the panel was opened from.
const URL_DEFAULTS = {
  range: ACTIVITY_DEFAULT_KEY,
  start_date: "",
  end_date: "",
  q: "",
  level: "",
  page: "0",
  size: String(DEFAULT_PAGE_SIZE),
  peek: "",
  peek_workspace: "",
} as const
const LEVEL_OPTIONS = [
  { value: "", label: "All levels" },
  { value: "failed", label: "Errors" },
]
const MINUTE_MS = 60_000
// Where "All" starts its chart when nothing has been recorded yet.
const ALL_FALLBACK_MS = 30 * 24 * 3_600_000

const EMPTY = <span className="text-subtle">—</span>

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`

function columns(peekId: string): DataTableColumn<TraceSummary>[] {
  return [
    {
      id: "last_activity",
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
  const url = useUrlState(URL_DEFAULTS)
  const { selected: workspace } = useSelectedWorkspace()
  const isMobile = useIsMobile()
  const [now, setNow] = useState(() => Date.now())

  const query = url.get("q")
  const level = url.get("level")
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
          <FilterChips
            chips={[]}
            start={
              <SearchField
                label="Search sessions"
                placeholder="Search by session ID"
                value={query}
                onChange={(value) => url.patch({ q: value, page: 0 })}
                className="w-72 max-w-full"
              />
            }
            end={<RefreshButton onRefresh={refresh} />}
          >
            <FilterSelect
              ariaLabel="Level"
              label="Level"
              value={level}
              onChange={(value) => url.patch({ level: value, page: 0 })}
              options={LEVEL_OPTIONS}
            />
          </FilterChips>

          <TableScrollFrame className="otari-sessions-table">
            <DataTable
              ariaLabel="Sessions"
              columns={tableColumns}
              rows={rows}
              getRowKey={(session) => session.trace_id}
              isLoading={traces.isPending && !traces.data}
              emptyContent="No sessions in this window. A request joins a session when its client names one; otherwise it is a session of its own."
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
          onClose={() => url.patch({ peek: "", peek_workspace: "" })}
          onPrevious={previous ? () => openPeek(previous) : undefined}
          onNext={next ? () => openPeek(next) : undefined}
        />
      ) : null}
    </div>
  )
}
