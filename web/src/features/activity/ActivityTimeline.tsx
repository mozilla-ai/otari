import { Button, Spinner } from "@heroui/react"
import type { ReactNode } from "react"
import { useRef, useState } from "react"

import type { UsageBucket } from "@/client"
import {
  ChartLegend,
  type SeriesDef,
  type StackedPoint,
  TrendChart,
} from "@/shared/components/charts"
import {
  bucketDurationMs,
  bucketIndexRange,
  formatWindowLabel,
  type RangePreset,
  rangeFromBuckets,
} from "@/shared/helpers/timeRange"

// A request-volume histogram that doubles as the time-range selector for the
// Activity page (the context strip pattern CloudWatch Logs Insights, Kibana,
// and Grafana Loki use for log exploration). Presets set the *extent* the strip
// spans; **dragging across the chart selects the window** (the standard
// interaction everywhere from Grafana to the OpenAI usage dashboard), regions
// outside the window dim in place, and −/+ buttons step the zoom (zooming out
// past the extent promotes to the next larger preset, so a wrong default is one
// tap away from a wider view). Failed requests render as a red segment on top
// of each bar, so dropped traffic is visible right on the strip. A slim rail
// under the chart pans the zoomed window (pointer drag or arrow keys) without
// resizing it. Buckets are UTC-aligned, so the selection lines up with the bars
// and the caption reads in UTC.

export interface TimelinePoint {
  bucketStart: string
  requests: number
  errors?: number
}

interface ActivityTimelineProps {
  presets: RangePreset[]
  // The preset defining the current extent (highlighted). Stays highlighted even
  // while a sub-window is selected, since it still describes how far back we span.
  extentKey: string
  onPreset: (preset: RangePreset) => void
  // A selected sub-window, resolved to absolute instants (end exclusive).
  onSelectRange: (startIso: string, endIso: string) => void
  // The selection was widened back to the whole extent: fall back to the rolling
  // preset window rather than a bounded snapshot.
  onSelectFull: () => void
  series: TimelinePoint[]
  bucket: UsageBucket
  // The active window (server-echoed): positions the dimming and drives the caption.
  windowStart?: string
  windowEnd?: string
  loading?: boolean
  ariaLabel?: string
  action?: ReactNode
}

function formatTick(iso: string, bucket: UsageBucket): string {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return bucket === "hour"
    ? d.toLocaleTimeString(undefined, {
        hour: "2-digit",
        minute: "2-digit",
        timeZone: "UTC",
      })
    : d.toLocaleDateString(undefined, {
        month: "short",
        day: "numeric",
        timeZone: "UTC",
      })
}

const SUCCESS_SERIES: SeriesDef = {
  key: "success",
  label: "Succeeded",
  color: "var(--color-primary)",
}
const ERROR_SERIES: SeriesDef = {
  key: "errors",
  label: "Failed",
  color: "var(--color-danger)",
}
const PLAIN_SERIES: SeriesDef = {
  key: "requests",
  label: "Requests",
  color: "var(--color-primary)",
}

export function ActivityTimeline({
  presets,
  extentKey,
  onPreset,
  onSelectRange,
  onSelectFull,
  series,
  bucket,
  windowStart,
  windowEnd,
  loading = false,
  ariaLabel = "Request volume over the selected window",
  action,
}: ActivityTimelineProps) {
  const starts = series.map((p) => p.bucketStart)
  const n = series.length
  const label = formatWindowLabel(windowStart, windowEnd)

  // Errors stack only when the window actually has any, so the everyday strip
  // stays a single calm series and red keeps its "something failed" meaning.
  const hasErrors = series.some((p) => (p.errors ?? 0) > 0)
  const chartSeries = hasErrors
    ? [SUCCESS_SERIES, ERROR_SERIES]
    : [PLAIN_SERIES]
  const data: StackedPoint[] = series.map((p): StackedPoint => {
    const errors = Math.min(p.errors ?? 0, p.requests)
    return hasErrors
      ? { x: p.bucketStart, success: p.requests - errors, errors }
      : { x: p.bucketStart, requests: p.requests }
  })

  // The active window as inclusive bucket indices of the extent series. The pan
  // rail keeps a local copy only while a drag is in flight (live feedback);
  // otherwise the props are the single source of truth.
  const windowIdx =
    n > 0
      ? bucketIndexRange(starts, windowStart, windowEnd)
      : { startIndex: 0, endIndex: 0 }
  const [panSel, setPanSel] = useState<{
    startIndex: number
    endIndex: number
  } | null>(null)
  // Ref mirror for the pointer handlers: pointerup can fire before the last
  // pointermove's setState has re-rendered, and committing from the stale
  // closure would pan to the previous position.
  const panSelRef = useRef(panSel)
  const setPan = (next: { startIndex: number; endIndex: number } | null) => {
    panSelRef.current = next
    setPanSel(next)
  }
  const sel = panSel ?? windowIdx
  const span = sel.endIndex - sel.startIndex + 1
  const atFullExtent = sel.startIndex === 0 && sel.endIndex >= n - 1
  const zoomed = n > 0 && !atFullExtent

  // Commit an inclusive bucket range: the full extent falls back to the rolling
  // preset window; anything narrower resolves to absolute instants.
  const commit = (startIndex: number, endIndex: number) => {
    if (n === 0) return
    const lo = Math.max(0, Math.min(startIndex, endIndex))
    const hi = Math.min(n - 1, Math.max(startIndex, endIndex))
    if (lo === 0 && hi === n - 1) {
      onSelectFull()
      return
    }
    const range = rangeFromBuckets(starts, lo, hi, bucket)
    if (range) onSelectRange(range.startIso, range.endIso)
  }

  // Step zoom. Out doubles the window around its center; at the full extent it
  // promotes to the next larger preset, so "the default window is too narrow"
  // is always one tap from a wider view. In halves it (min one bucket). When the
  // extent is not one of the presets (a drill-down window from another page),
  // fall back to the smallest preset that broadens it, so zoom-out never dead-ends.
  const extentIndex = presets.findIndex((p) => p.key === extentKey)
  const extentSeconds =
    extentIndex >= 0
      ? presets[extentIndex].seconds
      : (n * bucketDurationMs(bucket)) / 1000
  const largerPreset =
    extentIndex >= 0
      ? presets[extentIndex + 1]
      : presets.find(
          (p) =>
            p.seconds === null ||
            (extentSeconds !== null && p.seconds > extentSeconds),
        )

  const applySpan = (newSpan: number) => {
    const target = Math.max(1, Math.min(n, Math.round(newSpan)))
    const center = (sel.startIndex + sel.endIndex + 1) / 2
    let lo = Math.round(center - target / 2)
    lo = Math.max(0, Math.min(n - target, lo))
    commit(lo, lo + target - 1)
  }

  const zoomOut = () => {
    if (atFullExtent) {
      if (largerPreset) onPreset(largerPreset)
      return
    }
    applySpan(span * 2)
  }
  const zoomIn = () => applySpan(span / 2)

  // Pan rail: slides the zoomed window without resizing it. Pointer drags with
  // capture and commits on release; arrow keys step one bucket, PageUp/Down a
  // whole window, Home/End to the extent edges.
  const railRef = useRef<HTMLDivElement>(null)
  const panStart = useRef<{ x: number; startIndex: number } | null>(null)

  const panTo = (
    startIndex: number,
  ): { startIndex: number; endIndex: number } => {
    const lo = Math.max(0, Math.min(n - span, startIndex))
    return { startIndex: lo, endIndex: lo + span - 1 }
  }

  const onPanKeyDown = (event: React.KeyboardEvent) => {
    const delta =
      event.key === "ArrowRight" || event.key === "ArrowUp"
        ? 1
        : event.key === "ArrowLeft" || event.key === "ArrowDown"
          ? -1
          : event.key === "PageUp"
            ? span
            : event.key === "PageDown"
              ? -span
              : event.key === "Home"
                ? -n
                : event.key === "End"
                  ? n
                  : 0
    if (delta === 0) return
    event.preventDefault()
    const next = panTo(sel.startIndex + delta)
    if (next.startIndex !== sel.startIndex)
      commit(next.startIndex, next.endIndex)
  }

  const onPanMove = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!panStart.current || !railRef.current) return
    const width = railRef.current.getBoundingClientRect().width
    if (width <= 0) return
    const dx = Math.round(((event.clientX - panStart.current.x) / width) * n)
    setPan(panTo(panStart.current.startIndex + dx))
  }

  const endPan = (event: React.PointerEvent<HTMLDivElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }
    panStart.current = null
    const committed = panSelRef.current
    setPan(null)
    if (committed && committed.startIndex !== windowIdx.startIndex) {
      commit(committed.startIndex, committed.endIndex)
    }
  }

  const loPct = n ? (sel.startIndex / n) * 100 : 0
  const hiPct = n ? ((sel.endIndex + 1) / n) * 100 : 100
  const panMax = Math.max(0, n - span)

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        {presets.map((preset) => (
          <Button
            key={preset.key}
            size="sm"
            variant={extentKey === preset.key ? "primary" : "outline"}
            onPress={() => onPreset(preset)}
          >
            {preset.label}
          </Button>
        ))}
        <div className="ml-auto flex items-center gap-3">
          <span className="text-xs text-muted">Showing {label} · UTC</span>
          {action}
        </div>
      </div>

      <div className="rounded-xl border border-border bg-surface p-2">
        <div className="flex items-center justify-between gap-2 px-1 pb-1">
          <span className="flex items-center gap-3">
            <span className="text-[11px] font-medium uppercase tracking-wide text-muted">
              Requests / {bucket === "hour" ? "hour" : "day"}
            </span>
            <ChartLegend series={chartSeries} />
          </span>
          <div className="flex items-center gap-1.5">
            <span className="hidden text-[11px] text-muted sm:inline">
              drag across the chart to zoom
            </span>
            <Button
              size="sm"
              variant="ghost"
              isIconOnly
              aria-label="Zoom in"
              isDisabled={n === 0}
              onPress={zoomIn}
            >
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                className="h-4 w-4"
                aria-hidden="true"
              >
                <path d="M12 5v14M5 12h14" strokeLinecap="round" />
              </svg>
            </Button>
            <Button
              size="sm"
              variant="ghost"
              isIconOnly
              aria-label="Zoom out"
              isDisabled={n === 0 || (atFullExtent && !largerPreset)}
              onPress={zoomOut}
            >
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                className="h-4 w-4"
                aria-hidden="true"
              >
                <path d="M5 12h14" strokeLinecap="round" />
              </svg>
            </Button>
            {zoomed ? (
              <Button
                size="sm"
                variant="ghost"
                onPress={() => commit(0, n - 1)}
              >
                Reset
              </Button>
            ) : null}
          </div>
        </div>

        {loading && n === 0 ? (
          <div className="flex h-[90px] items-center justify-center">
            <Spinner size="sm" />
          </div>
        ) : n === 0 ? (
          <div className="flex h-[90px] items-center justify-center text-xs text-muted">
            No activity in this range.
          </div>
        ) : (
          <div className="flex flex-col gap-1">
            <TrendChart
              data={data}
              series={chartSeries}
              formatValue={(value) => value.toLocaleString()}
              formatXTick={(iso) => formatTick(iso, bucket)}
              ariaLabel={ariaLabel}
              height={90}
              onSelectRange={commit}
              window={zoomed || panSel ? sel : null}
            />
            {/* Pan rail: a minimap-style scrollbar for the zoomed window. Only
                rendered while zoomed (at the full extent there is nothing to
                pan), so it never takes space or a tab stop otherwise. */}
            {zoomed || panSel ? (
              <div
                ref={railRef}
                className="relative h-2.5 w-full rounded-full bg-surface-alt"
              >
                <div
                  role="slider"
                  aria-label="Pan the selected window"
                  aria-valuemin={0}
                  aria-valuemax={panMax}
                  aria-valuenow={Math.min(sel.startIndex, panMax)}
                  aria-valuetext={`Window starting at ${formatTick(starts[sel.startIndex] ?? starts[0], bucket)}`}
                  tabIndex={0}
                  className="absolute inset-y-0 cursor-grab touch-none rounded-full bg-accent/40 outline-none hover:bg-accent/60 focus-visible:ring-2 focus-visible:ring-accent active:cursor-grabbing"
                  style={{
                    left: `${loPct}%`,
                    width: `${Math.max(2, hiPct - loPct)}%`,
                  }}
                  onKeyDown={onPanKeyDown}
                  onPointerDown={(event) => {
                    event.preventDefault()
                    panStart.current = {
                      x: event.clientX,
                      startIndex: sel.startIndex,
                    }
                    setPan({ ...sel })
                    event.currentTarget.setPointerCapture(event.pointerId)
                  }}
                  onPointerMove={onPanMove}
                  onPointerUp={endPan}
                  onPointerCancel={endPan}
                />
              </div>
            ) : null}
          </div>
        )}
      </div>
    </div>
  )
}
