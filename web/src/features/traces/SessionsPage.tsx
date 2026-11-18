import { useEffect, useMemo, useRef, useState } from "react"
import type { SortDescriptor } from "react-aria-components"

import type { TraceDetail, TraceSortKey } from "@/client"
import { RefreshButton } from "@/design-system/actions/RefreshButton"
import { DataTable } from "@/design-system/data/DataTable"
import {
  PAGE_SIZE_OPTIONS,
  TablePagination,
} from "@/design-system/data/TablePagination"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { SearchField } from "@/design-system/forms/SearchField"
import { useIsMobile } from "@/design-system/hooks/useIsMobile"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import { ActivityTimeline } from "@/features/activity/ActivityTimeline"
import { useObserveScope } from "@/features/observe/useObserveScope"
import { useOrganizationContext } from "@/shared/api/organizations"
import {
  DEFAULT_TRACE_SORT,
  type TraceFilters,
  useTrace,
  useTraceContentPolicy,
  useTraceCount,
  useTraceSeries,
  useTraces,
} from "@/shared/api/traces"
import { useWorkspaces } from "@/shared/api/workspaces"
import { anyHarnessMark } from "@/shared/components/marks/BrandMark"
import {
  ACTIVITY_DEFAULT_KEY,
  bucketDurationMs,
  CUSTOM_KEY,
  findPreset,
  SESSION_PRESETS,
} from "@/shared/helpers/timeRange"
import { useUrlState } from "@/shared/helpers/urlState"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"
import { useDebounced } from "@/shared/hooks/useDebounced"
import { TELEMETRY_EVENTS } from "@/shared/telemetry/events"
import { useTelemetry } from "@/shared/telemetry/overlayTelemetry"
import { BreakGlassLog } from "./BreakGlassLog"
import { exceptionalAccess } from "./exceptionalAccess"
import { toTrace } from "./fromApi"
import { SessionContent, ShareSessionAction } from "./SessionContent"
import { columns } from "./sessionColumns"
import { TracePeek } from "./TracePeek"

const DEFAULT_PAGE_SIZE = 50

// The list's filters, window and page live in the URL beside the open session
// and its picked span, so a shared link or Back returns to the list the panel
// was opened from, and a link from Activity opens on one LLM call. `status`,
// `model`, `user_id` and `api_key_id` are the keys the Overview tabs share.
const URL_DEFAULTS = {
  range: ACTIVITY_DEFAULT_KEY,
  start_date: "",
  end_date: "",
  q: "",
  status: "",
  model: "",
  user_id: "",
  api_key_id: "",
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

// What a status filter means for a session: one with an unrecovered error, or
// one without. Any other request status has no session counterpart.
const HAS_ERROR: Record<string, "true" | "false"> = {
  error: "true",
  success: "false",
}

interface TimeWindow {
  start: number
  end: number
}

const iso = (ms: number) => new Date(ms).toISOString()

/**
 * The count the chart's header reads out: the server's count of the window.
 * One that could not be read leaves the header to the bars, rather than to
 * skeletons that never resolve.
 */
export function windowTotals(count: {
  data?: { count: number }
  isError: boolean
}): { count: number; failed: null } | null | undefined {
  if (count.data) return { count: count.data.count, failed: null }
  return count.isError ? undefined : null
}

/** Inside the Overview the page waits for its scope, so it never fetches the wrong one. */
export function SessionsPage({ embedded = false }: { embedded?: boolean }) {
  const { isResolved } = useObserveScope(embedded)
  return isResolved ? (
    <SessionsPageContent embedded={embedded} />
  ) : (
    <PageLoading />
  )
}

function SessionsPageContent({ embedded }: { embedded: boolean }) {
  const url = useUrlState(URL_DEFAULTS, { untracked: UNTRACKED })
  const { selected: workspace } = useSelectedWorkspace()
  // Said on the page as well as in the workspace's settings, because a manager
  // turns it on and a member may never have opted in themselves.
  const workspaces = useWorkspaces()
  const { policy } = useTraceContentPolicy()
  const platformBreakGlass = policy.platform_break_glass
  const keeping = policy.content_available
    ? (workspaces.data ?? []).filter(
        (candidate) =>
          candidate.trace_content_enabled &&
          (workspace ? candidate.id === workspace.workspace_id : true),
      )
    : []
  const keepsContent = keeping.length > 0
  // A workspace can stop keeping tool output that the deployment keeps.
  const keepsToolOutput =
    policy.tool_output &&
    keeping.some((candidate) => candidate.trace_content_tool_output ?? true)
  const isMobile = useIsMobile()
  const [now, setNow] = useState(() => Date.now())
  // The window a timeline selection asked for, until its list lands.
  const pendingSelection = useRef("")
  const { recordEvent } = useTelemetry()
  // The session this page is asking to open, and how. Kept with the session it
  // names, so a navigation that lands somewhere else (Back overtaking a press,
  // say) is not credited to the press. Anything else that opens one came by a
  // link.
  const opening = useRef<{ id: string; source: "list" | "timeline" } | null>(
    null,
  )

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
  // A start alone (Usage's Today, the attention strip's link) runs to now. A
  // bound that does not parse (a hand-edited link) falls back to the preset.
  const customStart = startParam ? Date.parse(startParam) : Number.NaN
  const customEnd = endParam ? Date.parse(endParam) : now
  const custom: TimeWindow | undefined =
    Number.isFinite(customStart) && Number.isFinite(customEnd)
      ? { start: customStart, end: customEnd }
      : undefined

  const range = url.get("range")
  const patch = url.patch
  // A `range` carried over from a page with other presets (Usage's 90d) reads
  // as the default window, so the URL is corrected to say so. Explicit bounds
  // win over a range, which is then not read.
  useEffect(() => {
    if (startParam || endParam) return
    if (range === CUSTOM_KEY || findPreset(SESSION_PRESETS, range)) return
    patch({ range: ACTIVITY_DEFAULT_KEY })
  }, [range, startParam, endParam, patch])

  const statusFilter = url.get("status")
  const modelFilters = url.getAll("model")
  // Inside the Overview the scope decides this; elsewhere it is the URL's own.
  const { userIds: userFilters } = useObserveScope(embedded)
  const apiKeyFilters = url.getAll("api_key_id")
  const hasError = HAS_ERROR[statusFilter] ?? ""

  const extent =
    findPreset(SESSION_PRESETS, range) ??
    findPreset(SESSION_PRESETS, ACTIVITY_DEFAULT_KEY)
  const bucket = extent?.bucket ?? "hour"
  const extentSeconds = extent?.seconds ?? null
  // Rounded down to the minute, so the query key holds still between renders.
  const extentStart =
    extentSeconds === null
      ? null
      : Math.floor((now - extentSeconds * 1000) / MINUTE_MS) * MINUTE_MS

  const search = useDebounced(query.trim().slice(0, SEARCH_MAX_LENGTH))
  const base: Omit<TraceFilters, "start"> = {
    workspaceId: workspace?.workspace_id ?? "",
    q: search,
    userIds: userFilters,
    apiKeyIds: apiKeyFilters,
    models: modelFilters,
    hasError,
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
  useEffect(() => {
    // Any change of the open session settles the press, whether it opened that
    // session, another one or none: Back overtaking it can leave the panel
    // closed, and a later link to the same session is then a link.
    const pressed = opening.current
    opening.current = null
    if (!peekId) return
    const source = pressed?.id === peekId ? pressed.source : "link"
    recordEvent(TELEMETRY_EVENTS.SESSION_OPENED, { source })
  }, [peekId, recordEvent])
  // An owner of the organization may open a private session by break-glass. What
  // it opened is held here for the session it opened, never polled over, and
  // dropped once the peek shows anything else or closes, so seeing it again takes
  // a new opening, which is recorded again.
  const isOwner = useOrganizationContext().data?.role === "owner"
  const [opened, setOpened] = useState<TraceDetail>()
  const peekWorkspace = url.get("peek_workspace")
  const openedHere =
    opened &&
    opened.summary.trace_id === peekId &&
    (!peekWorkspace || opened.summary.workspace_id === peekWorkspace)
      ? opened
      : undefined
  if (opened && !openedHere) setOpened(undefined)
  const shown = openedHere ?? detail.data

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

  const rows = traces.data?.items ?? []
  const withMarks = anyHarnessMark(rows)
  const tableColumns = useMemo(
    () => columns(peekId, withMarks),
    [peekId, withMarks],
  )
  const openIndex = rows.findIndex((session) => session.trace_id === peekId)
  // The read's own summary once it lands: the list is a snapshot, and a
  // session still running has moved on since. Until then, the row on screen.
  const openSummary =
    shown?.summary.trace_id === peekId ? shown.summary : rows[openIndex]
  const open = useMemo(
    () => (openSummary ? toTrace(openSummary, shown) : null),
    [openSummary, shown],
  )
  // A trace is keyed by its workspace and its id, so the URL keeps both.
  const openPeek = (id: string) => {
    // The open one pressed again changes nothing, so nothing is recorded.
    if (id !== peekId) opening.current = { id, source: "list" }
    url.patch({
      peek: id,
      peek_workspace:
        rows.find((session) => session.trace_id === id)?.workspace_id ?? "",
      span: "",
    })
  }
  const previous = openIndex > 0 ? rows[openIndex - 1].trace_id : null
  const next =
    openIndex >= 0 && openIndex < rows.length - 1
      ? rows[openIndex + 1].trace_id
      : null

  // A bar picked on the timeline that holds one session opens it. Only the
  // list answering that selection counts: a failed read or a filter change
  // drops it, so a later list never opens a session nobody picked.
  const listed = traces.data?.items
  const settled = !traces.isFetching && !traces.isPlaceholderData
  const filterKey = JSON.stringify(base)
  const isListError = traces.isError
  useEffect(() => {
    if (!pendingSelection.current) return
    if (isListError || !pendingSelection.current.endsWith(`|${filterKey}`)) {
      pendingSelection.current = ""
      return
    }
    if (!listed || !settled) return
    if (!pendingSelection.current.startsWith(`${startParam}|${endParam}|`)) {
      return
    }
    pendingSelection.current = ""
    if (listed.length !== 1) return
    // A bar whose one session is already open opens nothing new.
    if (listed[0].trace_id !== peekId) {
      opening.current = { id: listed[0].trace_id, source: "timeline" }
    }
    patch({
      peek: listed[0].trace_id,
      peek_workspace: listed[0].workspace_id,
      span: "",
    })
  }, [
    listed,
    settled,
    isListError,
    filterKey,
    startParam,
    endParam,
    patch,
    peekId,
  ])

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
        {embedded ? null : <PageIntro title="Sessions" />}
        {keepsContent ? (
          <InfoBanner>
            {workspace ? "This workspace keeps" : "Your workspaces keep"} what
            your sessions were about: the prompts and tool calls your agents
            send{keepsToolOutput ? ", and what each tool returned" : ""}. Otari
            shows yours only to you, unless you share a session.{" "}
            {exceptionalAccess(platformBreakGlass)}
          </InfoBanner>
        ) : null}

        <div className="flex flex-col gap-3">
          {traces.isError && !traces.data ? (
            <ErrorBanner error={traces.error} />
          ) : null}
          {/* With no row to open on, the panel cannot say it either. */}
          {detail.isError && !open ? (
            <ErrorBanner error={detail.error} />
          ) : null}
          <ActivityTimeline
            presets={SESSION_PRESETS}
            hidePresets={embedded}
            extentKey={
              custom ? CUSTOM_KEY : (extent?.key ?? ACTIVITY_DEFAULT_KEY)
            }
            onPreset={(preset) => {
              pendingSelection.current = ""
              url.patch({
                range: preset.key,
                start_date: "",
                end_date: "",
                page: 0,
              })
            }}
            onSelectRange={(startIso, endIso) => {
              pendingSelection.current = `${startIso}|${endIso}|${filterKey}`
              url.patch({ start_date: startIso, end_date: endIso, page: 0 })
            }}
            onSelectFull={() => {
              pendingSelection.current = ""
              url.patch({ start_date: "", end_date: "", page: 0 })
            }}
            series={series}
            bucket={bucket}
            windowStart={iso(win.start)}
            windowEnd={iso(win.end)}
            loading={seriesQuery.isPending}
            ariaLabel="Sessions active over the selected window"
            unitLabel="Active sessions"
            // A session spans buckets, so the bars are never summed; the
            // window's count is the one that sizes the list. No failed figure:
            // the series counts sessions with a failing error, and the
            // `has_error` filter counts any error, so a count with it would not
            // match the failed marks.
            additive={false}
            totals={windowTotals(count)}
          />
          <div className="otari-toolbar flex flex-wrap items-center gap-2">
            <SearchField
              label="Search sessions"
              placeholder="Search ID, name, agent or user"
              value={query}
              onChange={(value) => url.patch({ q: value, page: 0 })}
              maxLength={SEARCH_MAX_LENGTH}
              // The row's width on a phone, short of Refresh, so the
              // placeholder fits.
              className="w-72 max-w-full max-sm:w-auto max-sm:min-w-0 max-sm:flex-1"
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
          {isOwner ? <BreakGlassLog /> : null}
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
          notice={
            shown ? (
              <SessionContent
                detail={shown}
                onBreakGlass={isOwner ? setOpened : undefined}
              />
            ) : null
          }
          actions={shown ? <ShareSessionAction detail={shown} /> : null}
        />
      ) : null}
    </div>
  )
}
