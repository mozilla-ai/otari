import { Button } from "@heroui/react"
import type { ReactNode } from "react"
import { useRef, useState, useSyncExternalStore } from "react"
import { FiMinus, FiPlus } from "react-icons/fi"

import { IconButton } from "@/design-system/actions/IconButton"
import { Tab, TabRow } from "@/design-system/navigation/TabRow"
import {
  BUCKET_LABEL,
  bucketDurationMs,
  bucketIndexRange,
  type ChartBucket,
  formatWindowLabel,
  type RangePreset,
  rangeFromBuckets,
} from "@/shared/helpers/timeRange"
import { ChartReadout } from "./ChartReadout"
import { pointedReading, windowReading } from "./chartMath"
import { type Pointed, StripChart } from "./StripChart"

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

interface ActivityTimelineProps<B extends ChartBucket> {
  presets: RangePreset<B>[]
  // The preset defining the current extent (highlighted). Stays highlighted even
  // while a sub-window is selected, since it still describes how far back we span.
  extentKey: string
  onPreset: (preset: RangePreset<B>) => void
  // The presets are picked elsewhere (the Overview's filter bar), so the row
  // drops them; they still bound how far the chart zooms out.
  hidePresets?: boolean
  // A selected sub-window, resolved to absolute instants (end exclusive).
  onSelectRange: (startIso: string, endIso: string) => void
  // The selection was widened back to the whole extent: fall back to the rolling
  // preset window rather than a bounded snapshot.
  onSelectFull: () => void
  series: TimelinePoint[]
  bucket: ChartBucket
  // The active window (server-echoed): positions the dimming and drives the caption.
  windowStart?: string
  windowEnd?: string
  loading?: boolean
  ariaLabel?: string
  action?: ReactNode
  // What a bar counts, in the heading and the tooltip.
  unitLabel?: string
  // The count the window holds, from the caller. Null while it loads; left
  // out, the totals are summed from the series when `additive`.
  totals?: { count: number; failed: number | null } | null
  // False when a bucket can hold what its neighbor holds (a session spans
  // days), so the series must never be summed into a total.
  additive?: boolean
}

// Below `md` the zoom buttons are 44px and the chart takes its phone geometry.
const PHONE_QUERY = "(max-width: 767px)"
function usePhone(): boolean {
  return useSyncExternalStore(
    (onChange) => {
      if (typeof window.matchMedia !== "function") return () => {}
      const query = window.matchMedia(PHONE_QUERY)
      query.addEventListener("change", onChange)
      return () => query.removeEventListener("change", onChange)
    },
    () =>
      typeof window.matchMedia === "function" &&
      window.matchMedia(PHONE_QUERY).matches,
    () => false,
  )
}

function formatTick(iso: string, bucket: ChartBucket): string {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  if (bucket === "day") {
    return d.toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      timeZone: "UTC",
    })
  }
  // A week of 6-hour buckets repeats every time of day, so each tick says which day too.
  return d.toLocaleString(undefined, {
    ...(bucket === "6h" ? { month: "short", day: "numeric" } : {}),
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "UTC",
  })
}

export function ActivityTimeline<B extends ChartBucket>({
  presets,
  extentKey,
  onPreset,
  hidePresets = false,
  onSelectRange,
  onSelectFull,
  series,
  bucket,
  windowStart,
  windowEnd,
  loading = false,
  ariaLabel = "Request volume over the selected window",
  action,
  unitLabel = "Requests",
  totals,
  additive = true,
}: ActivityTimelineProps<B>) {
  const phone = usePhone()
  const [pointed, setPointed] = useState<Pointed | null>(null)
  const starts = series.map((point) => point.bucketStart)
  const n = series.length
  const label = formatWindowLabel(windowStart, windowEnd)

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
  }>()
  // Ref mirror for the pointer handlers: pointerup can fire before the last
  // pointermove's setState has re-rendered, and committing from the stale
  // closure would pan to the previous position.
  const panSelRef = useRef(panSel)
  const setPan = (next?: { startIndex: number; endIndex: number }) => {
    panSelRef.current = next
    setPanSel(next)
  }
  const sel = panSel ?? windowIdx
  const span = sel.endIndex - sel.startIndex + 1
  const atFullExtent = sel.startIndex === 0 && sel.endIndex >= n - 1
  const isZoomed = n > 0 && !atFullExtent

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
  const extentIndex = presets.findIndex((preset) => preset.key === extentKey)
  const extentSeconds =
    extentIndex >= 0
      ? presets[extentIndex].seconds
      : (n * bucketDurationMs(bucket)) / 1000
  const largerPreset =
    extentIndex >= 0
      ? presets[extentIndex + 1]
      : presets.find(
          (preset) =>
            preset.seconds === null ||
            (extentSeconds !== null && preset.seconds > extentSeconds),
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
  // The rail's width is measured once, where the drag starts, rather than on
  // every move. `getBoundingClientRect()` flushes pending layout before it can
  // answer, and the line below it writes state that dirties layout again, so
  // reading it here made the whole drag a read-write cycle per pointer event.
  // The rail cannot resize while a pointer is down on it.
  const panStart = useRef<{
    x: number
    startIndex: number
    railWidth: number
  } | null>(null)

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
    if (!panStart.current) return
    const { x, startIndex, railWidth } = panStart.current
    if (railWidth <= 0) return
    const dx = Math.round(((event.clientX - x) / railWidth) * n)
    const next = panTo(startIndex + dx)
    // Same bucket as the last move: nothing to redraw.
    if (next.startIndex === panSelRef.current?.startIndex) return
    setPan(next)
  }

  const endPan = (event: React.PointerEvent<HTMLDivElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }
    panStart.current = null
    const committed = panSelRef.current
    setPan(undefined)
    if (committed && committed.startIndex !== windowIdx.startIndex) {
      commit(committed.startIndex, committed.endIndex)
    }
  }

  const loPct = n ? (sel.startIndex / n) * 100 : 0
  const hiPct = n ? ((sel.endIndex + 1) / n) * 100 : 100
  const panMax = Math.max(0, n - span)

  // What the header reads out. Totals the caller gives win;
  // otherwise they are summed over the active window, unless the series cannot
  // be summed, when the tallest bucket stands in.
  const buckets = series.map((point) => ({
    requests: point.requests,
    errors: point.errors ?? 0,
  }))
  const unitWord = unitLabel.toLowerCase()
  const windowTotals = windowReading({
    totals,
    additive,
    loading,
    series: buckets,
    startIndex: sel.startIndex,
    endIndex: sel.endIndex,
  })
  const pointedTotals = pointed
    ? pointedReading({
        base: windowTotals,
        additive,
        series: buckets,
        lo: pointed.lo,
        hi: pointed.hi,
      })
    : null
  const pointedLabel = pointed
    ? pointed.lo === pointed.hi
      ? formatTick(starts[pointed.lo], bucket)
      : `${formatTick(starts[pointed.lo], bucket)} \u2013 ${formatTick(starts[pointed.hi], bucket)}`
    : undefined
  const readout = (
    <ChartReadout
      reading={pointedTotals ?? windowTotals}
      label={pointedLabel}
      unit={unitWord}
    />
  )

  const toolbar = (
    <div className="flex items-center gap-1.5">
      <span className="hidden text-xs text-muted sm:inline">
        Click a bar or drag to filter
      </span>
      {/* `md:min-h-8 md:min-w-8` on both: 44px where a finger does the
                zooming, back to the toolbar's own 32px density from `md` up. */}
      <IconButton
        size="sm"
        variant="ghost"
        isIconOnly
        className="md:min-h-8 md:min-w-8"
        label="Zoom in"
        isDisabled={n === 0}
        onPress={zoomIn}
      >
        <FiPlus aria-hidden="true" className="h-4 w-4" />
      </IconButton>
      <IconButton
        size="sm"
        variant="ghost"
        isIconOnly
        className="md:min-h-8 md:min-w-8"
        label="Zoom out"
        isDisabled={n === 0 || (atFullExtent && !largerPreset)}
        onPress={zoomOut}
      >
        <FiMinus aria-hidden="true" className="h-4 w-4" />
      </IconButton>
      {isZoomed ? (
        <Button size="sm" variant="ghost" onPress={() => commit(0, n - 1)}>
          Reset
        </Button>
      ) : null}
    </div>
  )

  // Pan rail: a minimap-style scrollbar for the zoomed window. Only
  // rendered while zoomed (at the full extent there is nothing to
  // pan), so it never takes space or a tab stop otherwise.
  const panRail =
    isZoomed || panSel ? (
      // The row is 44px so the handle inside it is a real touch target;
      // the track and the handle keep the 10px they read best at, drawn
      // as children so the grab area is the whole height rather than
      // the stripe. A thumb the size of the stripe is unpannable on a
      // phone, which is the only place the zoom is hard to redo.
      <div
        // The row is 44px so the handle inside it is a real touch
        // target; the track and the handle keep the 10px they read best
        // at, drawn as children so the grab area is the whole height
        // rather than the stripe. A thumb the size of the stripe is
        // unpannable on a phone, which is the only place the zoom is
        // hard to redo.
        ref={railRef}
        className="relative flex h-11 w-full items-center"
      >
        <div
          aria-hidden="true"
          className="absolute inset-x-0 h-2.5 bg-surface-alt"
        />
        <div
          role="slider"
          aria-label="Pan the selected window"
          aria-valuemin={0}
          aria-valuemax={panMax}
          aria-valuenow={Math.min(sel.startIndex, panMax)}
          aria-valuetext={`Window starting at ${formatTick(starts[sel.startIndex] ?? starts[0], bucket)}`}
          tabIndex={0}
          // Square, and here that is the point rather than
          // consistency: the window's edges assert two exact instants,
          // and a rounded end blurs the only thing the shape states.
          //
          // The fill lives on the stripe inside, not here: this element
          // is the 44px grab target and the stripe is the 10px drawing,
          // and painting both stacked two 40% accents into one 64% one.
          // `group` so the stripe's own variants have the ancestor they
          // address; without it they matched nothing and the hover was
          // dead. No `outline-none`, so the base `:focus-visible` rule
          // still rings the element that actually takes focus.
          className="group absolute inset-y-0 flex cursor-grab touch-none items-center active:cursor-grabbing"
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
              railWidth: railRef.current?.getBoundingClientRect().width ?? 0,
            }
            setPan({ ...sel })
            event.currentTarget.setPointerCapture(event.pointerId)
          }}
          onPointerMove={onPanMove}
          onPointerUp={endPan}
          onPointerCancel={endPan}
        >
          <div className="h-2.5 w-full bg-accent/40 group-hover:bg-accent/60" />
        </div>
      </div>
    ) : null

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        {hidePresets ? null : (
          <TabRow>
            {presets.map((preset) => (
              <Tab
                key={preset.key}
                isActive={extentKey === preset.key}
                onPress={() => onPreset(preset)}
              >
                {preset.label}
              </Tab>
            ))}
          </TabRow>
        )}
        {/* Full width on a phone, so the line starts on the column edge. */}
        <div className="ml-auto flex items-center gap-3 max-md:ml-0 max-md:w-full max-md:justify-between">
          {/* "· UTC" stays with the date before it and the lines balance, so a
              narrow row never leaves "UTC" alone on a line. */}
          <span className="text-caption text-balance">
            Showing {label}
            {"\u00a0·\u00a0"}UTC
          </span>
          {action}
        </div>
      </div>

      {/* A band of the page between rules, not a card. The totals read out in
          the header; there is no legend. */}
      <div className="border-y border-border py-3">
        <div className="grid grid-cols-[1fr_auto] gap-x-2 gap-y-0.5 md:flex md:h-9 md:items-center md:gap-4">
          <span className="self-center whitespace-nowrap text-overline md:order-1">
            {unitLabel} / {BUCKET_LABEL[bucket]}
          </span>
          <div className="md:order-3 md:ml-auto">{toolbar}</div>
          <div className="col-span-2 flex min-h-5 items-center md:order-2 md:col-auto md:min-h-0">
            {readout}
          </div>
        </div>
        <div className="flex flex-col gap-1 pt-1">
          <StripChart
            data={buckets.map((point, index) => ({
              ...point,
              x: starts[index],
            }))}
            formatTick={(iso) => formatTick(iso, bucket)}
            ariaLabel={ariaLabel}
            window={isZoomed || panSel ? sel : undefined}
            onSelectRange={commit}
            onReading={setPointed}
            phone={phone}
            loading={loading}
            emptyLabel="No activity in this range."
          />
          {panRail}
        </div>
      </div>
    </div>
  )
}
