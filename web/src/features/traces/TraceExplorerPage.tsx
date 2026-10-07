import { useCallback, useMemo, useState } from "react"
import type { Selection } from "react-aria-components"
import { FiInfo } from "react-icons/fi"

import { RefreshButton } from "@/design-system/actions/RefreshButton"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { TablePagination } from "@/design-system/data/TablePagination"
import { SearchField } from "@/design-system/forms/SearchField"
import { Chip } from "@/design-system/indicators/Chip"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import { FilterChips } from "@/design-system/navigation/FilterChips"
import { FilterSelect } from "@/design-system/navigation/FilterSelect"
import { Tooltip } from "@/design-system/overlays/Tooltip"
import { ActivityTimeline } from "@/features/activity/ActivityTimeline"
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

import { formatDuration, formatTraceTimestamp } from "./format"
import { MOCK_TRACES } from "./mockData"
import { TracePeek } from "./TracePeek"
import { summarizeTrace, type Trace } from "./types"

const URL_DEFAULTS = { peek: "" } as const
const LEVEL_OPTIONS = [
  { value: "", label: "All levels" },
  { value: "ERROR", label: "Errors" },
  { value: "WARNING", label: "Warnings" },
]

const EMPTY = <span className="text-subtle">—</span>

function snippet(value: unknown): string | null {
  if (value == null) return null
  if (typeof value === "object" && "message" in value) {
    return String((value as { message: unknown }).message)
  }
  return JSON.stringify(value)
}

function TextCell({ value }: { value: unknown }) {
  const text = snippet(value)
  return text ? (
    <span className="block truncate text-mono-caption text-muted">{text}</span>
  ) : (
    EMPTY
  )
}

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`

const COLUMNS: DataTableColumn<Trace>[] = [
  {
    id: "timestamp",
    header: "Timestamp",
    cell: (t) => (
      <span className="text-mono-caption whitespace-nowrap text-muted">
        {formatTraceTimestamp(t.timestamp)}
      </span>
    ),
  },
  {
    id: "name",
    header: "Name",
    isRowHeader: true,
    cell: (t) => <span className="block truncate text-body">{t.name}</span>,
  },
  { id: "input", header: "Input", cell: (t) => <TextCell value={t.input} /> },
  {
    id: "output",
    header: "Output",
    cell: (t) => <TextCell value={t.output} />,
  },
  {
    id: "levels",
    header: "Levels",
    cell: (t) => {
      const { errorCount, warningCount } = summarizeTrace(t)
      if (!errorCount && !warningCount) return EMPTY
      return (
        <span className="flex gap-1 whitespace-nowrap">
          {errorCount ? (
            <Chip tone="danger">{plural(errorCount, "error")}</Chip>
          ) : null}
          {warningCount ? (
            <Chip tone="warning">{plural(warningCount, "warning")}</Chip>
          ) : null}
        </span>
      )
    },
  },
  {
    id: "latency",
    header: "Latency",
    align: "end",
    cell: (t) => (
      <span className="text-mono-caption">
        {formatDuration(summarizeTrace(t).latencyMs) ?? EMPTY}
      </span>
    ),
  },
  {
    id: "tokens",
    header: "Tokens",
    align: "end",
    cell: (t) => {
      const s = summarizeTrace(t)
      return s.totalTokens ? (
        <span className="text-mono-caption whitespace-nowrap">
          {formatNumber(s.promptTokens)} → {formatNumber(s.completionTokens)} (Σ{" "}
          {formatNumber(s.totalTokens)})
        </span>
      ) : (
        EMPTY
      )
    },
  },
  {
    id: "cost",
    header: "Total cost",
    align: "end",
    cell: (t) => {
      const s = summarizeTrace(t)
      if (!s.totalCost) return EMPTY
      return (
        <span className="inline-flex items-center justify-end gap-1 text-mono-caption">
          {formatCost(s.totalCost)}
          <Tooltip
            content={
              <span className="text-mono-caption">
                Input {formatCost(s.inputCost)} · Output{" "}
                {formatCost(s.outputCost)}
              </span>
            }
          >
            <FiInfo aria-hidden="true" className="h-3.5 w-3.5 text-subtle" />
          </Tooltip>
        </span>
      )
    },
  },
  {
    id: "tags",
    header: "Tags",
    cell: (t) =>
      t.tags.length ? (
        <span className="flex items-center gap-1">
          {t.tags.slice(0, 2).map((tag) => (
            <Chip key={tag}>{tag}</Chip>
          ))}
          {t.tags.length > 2 ? (
            <span className="text-caption text-subtle">
              +{t.tags.length - 2}
            </span>
          ) : null}
        </span>
      ) : (
        EMPTY
      ),
  },
]

interface Window {
  start: number
  end: number
}

export function TraceExplorerPage() {
  const url = useUrlState(URL_DEFAULTS)
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
  const extentWindow: Window = useMemo(
    () => ({
      start:
        extent.seconds === null
          ? Math.min(...MOCK_TRACES.map((t) => Date.parse(t.timestamp)))
          : now - extent.seconds * 1000,
      end: now,
    }),
    [extent, now],
  )
  const win = custom ?? extentWindow

  const series = useMemo(() => {
    const size = bucketDurationMs(extent.bucket)
    const first = Math.floor(extentWindow.start / size) * size
    const points = []
    for (let at = first; at < extentWindow.end; at += size) {
      const inBucket = MOCK_TRACES.filter((t) => {
        const ms = Date.parse(t.timestamp)
        return ms >= at && ms < at + size
      })
      points.push({
        bucketStart: new Date(at).toISOString(),
        requests: inBucket.length,
        errors: inBucket.filter((t) => summarizeTrace(t).errorCount > 0).length,
      })
    }
    return points
  }, [extent, extentWindow])

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase()
    return MOCK_TRACES.filter((t) => {
      const ms = Date.parse(t.timestamp)
      if (ms < win.start || ms >= win.end) return false
      if (level) {
        const s = summarizeTrace(t)
        if (level === "ERROR" ? !s.errorCount : !s.warningCount) return false
      }
      return (
        !q ||
        [t.id, t.name, t.userId, t.sessionId, ...t.tags].some((v) =>
          v.toLowerCase().includes(q),
        )
      )
    })
  }, [query, level, win])

  const pageRows = rows.slice(page * pageSize, (page + 1) * pageSize)

  const peekId = url.get("peek")
  const openIndex = rows.findIndex((t) => t.id === peekId)
  const open = openIndex >= 0 ? rows[openIndex] : null
  const openPeek = useCallback((id: string) => url.patch({ peek: id }), [url])

  const previous = openIndex > 0 ? rows[openIndex - 1].id : null
  const next =
    openIndex >= 0 && openIndex < rows.length - 1
      ? rows[openIndex + 1].id
      : null

  const onSelectRange = (startIso: string, endIso: string) => {
    setCustom({ start: Date.parse(startIso), end: Date.parse(endIso) })
    setPage(0)
  }

  return (
    <div className="flex flex-col">
      <PageIntro title="Activity">
        Every trace the gateway recorded: the calls, tools and routing attempts
        behind each agent session.
      </PageIntro>

      <div className="flex flex-col gap-3">
        <ActivityTimeline
          presets={ACTIVITY_PRESETS}
          extentKey={custom ? CUSTOM_KEY : extentKey}
          onPreset={(preset) => {
            setExtentKey(preset.key)
            setCustom(null)
            setPage(0)
          }}
          onSelectRange={onSelectRange}
          onSelectFull={() => setCustom(null)}
          series={series}
          bucket={extent.bucket}
          windowStart={new Date(win.start).toISOString()}
          windowEnd={new Date(win.end).toISOString()}
          ariaLabel="Trace volume over the selected window"
        />
        <FilterChips
          chips={[]}
          start={
            <SearchField
              label="Search traces"
              placeholder="Search by ID, name"
              value={query}
              onChange={(value) => {
                setQuery(value)
                setPage(0)
              }}
              className="w-72 max-w-full"
            />
          }
          end={<RefreshButton onRefresh={() => setNow(Date.now())} />}
        >
          <FilterSelect
            ariaLabel="Level"
            label="Level"
            value={level}
            onChange={(value) => {
              setLevel(value)
              setPage(0)
            }}
            options={LEVEL_OPTIONS}
          />
        </FilterChips>

        <TableScrollFrame className="otari-activity-table">
          <DataTable
            ariaLabel="Traces"
            columns={COLUMNS}
            rows={pageRows}
            getRowKey={(t) => t.id}
            emptyContent="No traces match these filters."
            selectionMode="multiple"
            selectedKeys={selection}
            onSelectionChange={setSelection}
            onRowAction={openPeek}
            rowClassName={(t) =>
              t.id === peekId ? "bg-surface-subtle" : undefined
            }
          />
        </TableScrollFrame>
        <TablePagination
          page={page}
          pageSize={pageSize}
          total={rows.length}
          rowsOnPage={pageRows.length}
          onPageChange={setPage}
          onPageSizeChange={(size) => {
            setPageSize(size)
            setPage(0)
          }}
        />
      </div>

      {open ? (
        <TracePeek
          trace={open}
          onClose={() => url.patch({ peek: "" })}
          onPrevious={previous ? () => openPeek(previous) : undefined}
          onNext={next ? () => openPeek(next) : undefined}
        />
      ) : null}
    </div>
  )
}
