import { useState } from "react"
import type { TraceDetail } from "@/client"
import { CopyButton } from "@/design-system/actions/CopyButton"
import { Chip } from "@/design-system/indicators/Chip"
import { Segmented } from "@/design-system/navigation/Segmented"
import { SpanDetails } from "@/features/traces/SpanDetails"
import { SpanTree } from "@/features/traces/SpanTree"
import {
  buildTraceTree,
  sessionDurationMs,
  sessionName,
  sessionSourceLabel,
  spanLabel,
  spanLog,
  spanOutcomeLabel,
} from "@/features/traces/traceModel"
import {
  formatCost,
  formatDateTime,
  formatLatency,
  formatTime,
  formatTokens,
} from "@/shared/helpers/format"

type View = "tree" | "log"

const VIEWS = [
  { value: "tree", label: "Preview" },
  { value: "log", label: "Log view" },
]

// One session: its totals, then its turns as a tree (or every span as a log), and
// the facts of whichever span is open. Stateless apart from which view and span
// are open, which belong to this panel and to nobody else.
export function TraceDetailPanel({ detail }: { detail: TraceDetail }) {
  const [view, setView] = useState<View>("tree")
  const [selectedSpanId, setSelectedSpanId] = useState("")
  const { summary } = detail
  const tree = buildTraceTree(detail)
  const selected =
    detail.spans.find((span) => span.span_id === selectedSpanId) ??
    detail.spans[0]
  return (
    <div className="flex min-w-0 flex-col gap-4 p-4">
      <header className="flex flex-col gap-2">
        <div className="flex min-w-0 items-center gap-2">
          <h3 className="truncate text-title">{sessionName(summary)}</h3>
          {detail.state === "active" ? (
            <Chip tone="info" size="sm">
              Active
            </Chip>
          ) : null}
        </div>
        <div className="flex min-w-0 items-center gap-2 text-caption">
          <span className="truncate font-mono">{summary.trace_id}</span>
          <CopyButton value={summary.trace_id} label="trace ID" />
        </div>
        <p className="flex flex-wrap gap-x-4 gap-y-1 text-caption tabular-nums">
          <span>{formatDateTime(summary.started_at)}</span>
          <span>{formatLatency(sessionDurationMs(summary))}</span>
          <span>{formatCost(summary.cost)}</span>
          <span>
            {formatTokens(summary.input_tokens + summary.output_tokens)} tokens
          </span>
          <span>{summary.step_count} requests</span>
        </p>
        <p className="text-caption">{sessionSourceLabel(summary)}</p>
        {detail.truncated ? (
          <p className="text-caption">
            This session has more spans than one view shows.
          </p>
        ) : null}
      </header>
      <Segmented
        label="View"
        options={VIEWS}
        value={view}
        onChange={(next) => setView(next as View)}
        size="sm"
      />
      <div className="grid min-w-0 grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="min-w-0">
          {view === "tree" ? (
            <SpanTree
              tree={tree}
              selectedSpanId={selected?.span_id ?? ""}
              onSelect={setSelectedSpanId}
            />
          ) : (
            <ol className="flex flex-col gap-1" aria-label="Spans in order">
              {spanLog(detail).map((span) => (
                <li key={span.span_id}>
                  <button
                    type="button"
                    onClick={() => setSelectedSpanId(span.span_id)}
                    aria-current={
                      span.span_id === selected?.span_id ? "true" : undefined
                    }
                    className={`flex min-h-11 w-full min-w-0 items-center gap-3 rounded-md px-2 text-left text-body ${
                      span.span_id === selected?.span_id
                        ? "bg-primary-subtle"
                        : "hover:bg-surface-alt"
                    }`}
                  >
                    <span className="w-20 shrink-0 text-caption tabular-nums">
                      {formatTime(span.start_time)}
                    </span>
                    <span className="min-w-0 flex-1 truncate text-body">
                      {spanLabel(span)}
                    </span>
                    <span className="shrink-0 text-caption">
                      {spanOutcomeLabel(span)}
                    </span>
                  </button>
                </li>
              ))}
            </ol>
          )}
        </div>
        <div className="min-w-0">
          {selected ? <SpanDetails span={selected} /> : null}
        </div>
      </div>
    </div>
  )
}
