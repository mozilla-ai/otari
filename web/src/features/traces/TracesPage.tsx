import type { ReactNode } from "react"
import type { TraceSummary } from "@/client"
import { CopyButton } from "@/design-system/actions/CopyButton"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { TablePagination } from "@/design-system/data/TablePagination"
import { EmptyMessage } from "@/design-system/feedback/EmptyMessage"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { SearchField } from "@/design-system/forms/SearchField"
import { Chip } from "@/design-system/indicators/Chip"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { SidePanel } from "@/design-system/layout/SidePanel"
import { Segmented } from "@/design-system/navigation/Segmented"
import { Tab, TabRow } from "@/design-system/navigation/TabRow"
import { TraceDetailPanel } from "@/features/traces/TraceDetailPanel"
import { TracesChart } from "@/features/traces/TracesChart"
import {
  sessionDurationMs,
  sessionFailed,
  sessionName,
  shortId,
  traceKind,
} from "@/features/traces/traceModel"
import {
  type TraceFilters,
  useTrace,
  useTraceCount,
  useTraceSeries,
  useTraces,
} from "@/shared/api/traces"
import {
  formatCost,
  formatDateTime,
  formatLatency,
  formatTokens,
} from "@/shared/helpers/format"
import {
  ACTIVITY_DEFAULT_KEY,
  ACTIVITY_PRESETS,
  findPreset,
} from "@/shared/helpers/timeRange"
import { useUrlState } from "@/shared/helpers/urlState"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"

const PAGE_SIZE = 50

const COLUMNS: DataTableColumn<TraceSummary>[] = [
  {
    id: "last_activity_at",
    header: "Last activity",
    cell: (summary) => (
      <span className="font-mono tabular-nums">
        {formatDateTime(summary.last_activity_at)}
      </span>
    ),
    minWidth: 170,
  },
  {
    id: "type",
    header: "Type",
    cell: (summary) => (
      <span className="text-caption">{traceKind(summary)}</span>
    ),
  },
  {
    id: "name",
    header: "Name",
    isRowHeader: true,
    cell: (summary) => (
      <span className="flex min-w-0 items-center gap-2">
        <span className="truncate">{sessionName(summary)}</span>
        <span className="shrink-0 font-mono text-caption">
          {shortId(summary.trace_id)}
        </span>
      </span>
    ),
    minWidth: 220,
  },
  {
    id: "steps",
    header: "Requests",
    align: "end",
    cell: (summary) => (
      <span className="tabular-nums">{summary.step_count}</span>
    ),
  },
  {
    id: "duration",
    header: "Duration",
    align: "end",
    cell: (summary) => (
      <span className="tabular-nums">
        {formatLatency(sessionDurationMs(summary))}
      </span>
    ),
  },
  {
    id: "tokens",
    header: "Tokens",
    align: "end",
    cell: (summary) => (
      <span className="tabular-nums">
        {formatTokens(summary.input_tokens + summary.output_tokens)}
      </span>
    ),
  },
  {
    id: "cost",
    header: "Cost",
    align: "end",
    cell: (summary) => (
      <span className="tabular-nums">{formatCost(summary.cost)}</span>
    ),
  },
  {
    id: "status",
    header: "Status",
    cell: (summary) =>
      sessionFailed(summary) ? (
        <Chip tone="danger" size="sm">
          Failed
        </Chip>
      ) : (
        <span className="text-caption">Completed</span>
      ),
  },
]
const STATUS_OPTIONS = [
  { value: "", label: "All" },
  { value: "failed", label: "Failed" },
]

const URL_DEFAULTS = {
  range: ACTIVITY_DEFAULT_KEY,
  q: "",
  outcome: "",
  trace: "",
  trace_workspace: "",
  session_page: "0",
}

// The window's start, rounded down to the minute so the query key holds still
// between renders instead of moving with the clock.
function windowStart(seconds: number | null): string {
  if (seconds === null) return ""
  const minute = 60_000
  return new Date(
    Math.floor((Date.now() - seconds * 1000) / minute) * minute,
  ).toISOString()
}

// Every agent session the gateway recorded, newest activity first; one opens in a
// panel over the page. The page owns the URL state and the queries; its parts take data.
export function TracesPage({ viewSwitch }: { viewSwitch: ReactNode }) {
  const url = useUrlState(URL_DEFAULTS)
  const preset =
    findPreset(ACTIVITY_PRESETS, url.get("range")) ??
    findPreset(ACTIVITY_PRESETS, ACTIVITY_DEFAULT_KEY)
  const { selected: workspace } = useSelectedWorkspace()
  const filters: TraceFilters = {
    // From the sidebar's switcher, as the request log scopes to it, so the two
    // views of Activity show the same workspace.
    workspaceId: workspace?.workspace_id ?? "",
    start: windowStart(preset?.seconds ?? null),
    q: url.get("q"),
    failedOnly: url.get("outcome") === "failed",
    harness: [],
  }
  const page = url.getNumber("session_page")
  const traceId = url.get("trace")

  const traces = useTraces(filters, page, PAGE_SIZE)
  const count = useTraceCount(filters)
  const series = useTraceSeries(
    filters,
    preset?.bucket === "day" ? "day" : "hour",
  )
  const trace = useTrace(traceId, url.get("trace_workspace"))

  const items = traces.data?.items ?? []
  // A trace is keyed by its workspace and its id, so the URL keeps both.
  const open = (id: string) =>
    url.patch({
      trace: id,
      trace_workspace:
        items.find((summary) => summary.trace_id === id)?.workspace_id ?? "",
    })
  // Previous and next walk the page of sessions on screen.
  const position = items.findIndex((summary) => summary.trace_id === traceId)
  const before = position > 0 ? items[position - 1] : undefined
  const after =
    position >= 0 && position < items.length - 1
      ? items[position + 1]
      : undefined
  // From the row on screen, or from the read when the URL opened one off this page.
  const selected = items[position] ?? trace.data?.summary
  const kind = selected ? traceKind(selected) : "Session"

  let detail: ReactNode = null
  if (trace.isError && !trace.data) {
    detail = (
      <div className="p-5">
        <ErrorBanner error={trace.error} />
      </div>
    )
  } else if (trace.isPending && !trace.data) {
    detail = <EmptyMessage>Loading the session…</EmptyMessage>
  } else if (trace.data) {
    detail = <TraceDetailPanel key={traceId} detail={trace.data} />
  }

  const isFirstLoad = traces.isPending && !traces.data
  return (
    <div className="flex min-w-0 flex-col gap-6">
      <PageIntro title="Activity" action={viewSwitch}>
        Every agent session the gateway recorded: the requests, LLM calls, tools
        and routing attempts behind each one.
      </PageIntro>
      {traces.isError && !traces.data ? (
        <ErrorBanner error={traces.error} />
      ) : null}
      <div className="flex flex-col gap-3">
        <TabRow>
          {ACTIVITY_PRESETS.map((option) => (
            <Tab
              key={option.key}
              isActive={option.key === preset?.key}
              onPress={() => url.patch({ range: option.key, session_page: 0 })}
            >
              {option.label}
            </Tab>
          ))}
        </TabRow>
        {series.data && series.data.points.length > 0 ? (
          <TracesChart series={series.data} />
        ) : null}
      </div>
      <div className="flex flex-col gap-3 md:flex-row md:items-end">
        <SearchField
          label="Search sessions"
          placeholder="Search by session ID"
          value={url.get("q")}
          onChange={(value) => url.patch({ q: value, session_page: 0 })}
          className="md:w-80"
        />
        <Segmented
          label="Status"
          options={STATUS_OPTIONS}
          value={url.get("outcome")}
          onChange={(value) => url.patch({ outcome: value, session_page: 0 })}
          size="sm"
        />
      </div>
      <DataTable
        ariaLabel="Sessions"
        columns={COLUMNS}
        rows={items}
        getRowKey={(summary) => summary.trace_id}
        isLoading={isFirstLoad}
        onRowAction={open}
        rowClassName={(summary) =>
          summary.trace_id === traceId ? "bg-primary-subtle" : undefined
        }
        emptyContent="Nothing in this window. Each request appears here, on its own or grouped into a session when its client sends a session id."
      />
      <SidePanel
        isOpen={traceId !== ""}
        onClose={() => url.patch({ trace: "", trace_workspace: "" })}
        label={kind}
        heading={
          <>
            <span className="text-overline">{kind}</span>
            <span className="truncate font-mono text-caption">{traceId}</span>
            <CopyButton value={traceId} label={`${kind.toLowerCase()} ID`} />
          </>
        }
        onPrevious={before ? () => open(before.trace_id) : undefined}
        onNext={after ? () => open(after.trace_id) : undefined}
      >
        {detail}
      </SidePanel>
      <TablePagination
        page={page}
        pageSize={PAGE_SIZE}
        total={count.data?.count ?? null}
        rowsOnPage={items.length}
        onPageChange={(next) => url.patch({ session_page: next })}
        onPageSizeChange={() => undefined}
        pageSizeOptions={[PAGE_SIZE]}
        isFetching={traces.isFetching}
        hasNextFallback={traces.data?.has_more ?? false}
        label="Sessions"
      />
    </div>
  )
}
