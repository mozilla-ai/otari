import type { ReactNode } from "react"
import { TablePagination } from "@/design-system/data/TablePagination"
import { EmptyMessage } from "@/design-system/feedback/EmptyMessage"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { SearchField } from "@/design-system/forms/SearchField"
import { Chip } from "@/design-system/indicators/Chip"
import { ListDetail, ListDetailRow } from "@/design-system/layout/ListDetail"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { Segmented } from "@/design-system/navigation/Segmented"
import { Tab, TabRow } from "@/design-system/navigation/TabRow"
import { TraceDetailPanel } from "@/features/traces/TraceDetailPanel"
import { TracesChart } from "@/features/traces/TracesChart"
import {
  sessionFailed,
  sessionName,
  shortId,
} from "@/features/traces/traceModel"
import {
  type TraceFilters,
  useTrace,
  useTraceCount,
  useTraceSeries,
  useTraces,
} from "@/shared/api/traces"
import { formatCost, formatDateTime } from "@/shared/helpers/format"
import {
  ACTIVITY_DEFAULT_KEY,
  ACTIVITY_PRESETS,
  findPreset,
} from "@/shared/helpers/timeRange"
import { useUrlState } from "@/shared/helpers/urlState"

const PAGE_SIZE = 50
const STATUS_OPTIONS = [
  { value: "", label: "All" },
  { value: "failed", label: "Failed" },
]

const URL_DEFAULTS = {
  range: ACTIVITY_DEFAULT_KEY,
  q: "",
  status: "",
  trace: "",
  page: "0",
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

// Every agent session the gateway recorded, newest activity first, beside the one
// that is open. The page owns the URL state and the queries; its parts take data.
export function TracesPage({ viewSwitch }: { viewSwitch: ReactNode }) {
  const url = useUrlState(URL_DEFAULTS)
  const preset =
    findPreset(ACTIVITY_PRESETS, url.get("range")) ??
    findPreset(ACTIVITY_PRESETS, ACTIVITY_DEFAULT_KEY)
  const filters: TraceFilters = {
    start: windowStart(preset?.seconds ?? null),
    q: url.get("q"),
    failedOnly: url.get("status") === "failed",
    harness: [],
  }
  const page = url.getNumber("page")
  const traceId = url.get("trace")

  const traces = useTraces(filters, page, PAGE_SIZE)
  const count = useTraceCount(filters)
  const series = useTraceSeries(
    filters,
    preset?.bucket === "day" ? "day" : "hour",
  )
  const trace = useTrace(traceId)

  const items = traces.data?.items ?? []
  const list = items.map((summary) => (
    <ListDetailRow
      key={summary.trace_id}
      isSelected={summary.trace_id === traceId}
      onSelect={() => url.patch({ trace: summary.trace_id })}
      label={
        <span className="flex min-w-0 items-center gap-2">
          <span className="truncate">{sessionName(summary)}</span>
          <span className="shrink-0 text-caption font-mono">
            {shortId(summary.trace_id)}
          </span>
          {sessionFailed(summary) ? (
            <Chip tone="danger" size="sm" className="shrink-0">
              Failed
            </Chip>
          ) : null}
        </span>
      }
    >
      <span className="flex flex-wrap gap-x-3 text-caption tabular-nums">
        <span>{formatDateTime(summary.last_activity_at)}</span>
        <span>{summary.step_count} requests</span>
        <span>{formatCost(summary.cost)}</span>
      </span>
    </ListDetailRow>
  ))

  let detail: ReactNode
  if (traceId === "") {
    detail = (
      <EmptyMessage>
        Open a session to see its turns, LLM calls and tool calls.
      </EmptyMessage>
    )
  } else if (trace.isError && !trace.data) {
    detail = <ErrorBanner error={trace.error} />
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
        and routing attempts behind each one. No prompt, output or tool content
        is stored.
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
              onPress={() => url.patch({ range: option.key, page: 0 })}
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
          onChange={(value) => url.patch({ q: value, page: 0 })}
          className="md:w-80"
        />
        <Segmented
          label="Status"
          options={STATUS_OPTIONS}
          value={url.get("status")}
          onChange={(value) => url.patch({ status: value, page: 0 })}
          size="sm"
        />
      </div>
      <ListDetail
        listLabel="Sessions"
        list={list}
        isEmpty={items.length === 0}
        empty={
          <EmptyMessage>
            {isFirstLoad
              ? "Loading sessions…"
              : "No sessions in this window. A client's requests are grouped into one session when it sends a session id."}
          </EmptyMessage>
        }
        detail={detail}
        detailLabel="Session"
        isDetailShown={traceId !== ""}
        onShowList={() => url.patch({ trace: "" })}
        backLabel="Back to the sessions"
      />
      <TablePagination
        page={page}
        pageSize={PAGE_SIZE}
        total={count.data?.count ?? null}
        rowsOnPage={items.length}
        onPageChange={(next) => url.patch({ page: next })}
        onPageSizeChange={() => undefined}
        pageSizeOptions={[PAGE_SIZE]}
        isFetching={traces.isFetching}
        hasNextFallback={traces.data?.has_more ?? false}
        label="Sessions"
      />
    </div>
  )
}
