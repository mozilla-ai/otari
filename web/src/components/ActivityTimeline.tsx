import { Button, Spinner } from "@heroui/react";
import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { Slider as RacSlider, SliderThumb as RacSliderThumb, SliderTrack as RacSliderTrack } from "react-aria-components";
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis } from "recharts";

import type { UsageBucket } from "@/api/types";
import { bucketDurationMs, bucketIndexRange, formatWindowLabel, rangeFromBuckets, type RangePreset } from "@/lib/timeRange";

// A request-volume histogram that doubles as the time-range selector for the
// Usage and Activity pages (the pattern CloudWatch Logs Insights, Kibana, and
// Grafana/Loki use for log exploration). Presets set the *extent* the chart
// spans; the selection lives on the chart itself: full-height handles at the
// window edges zoom, a slim strip along the axis pans, regions outside the
// window dim in place, and −/+ buttons step the zoom (zooming out past the
// extent promotes to the next larger preset, so a wrong default is one tap
// away from a wider view). The tiles and log below follow the selection.
//
// Smoothness: the handles are react-aria slider thumbs (unstyled primitives,
// the same machinery HeroUI wraps), so they are pointer-captured and glide at
// 0.1-bucket steps. Drag feedback is CSS-positioned shades; the recharts tree
// never re-renders mid-drag, and the window commits once, on release, snapped
// outward to whole buckets. Buckets are UTC-aligned, so the selection lines up
// with the bars and the caption reads in UTC.

export interface TimelinePoint {
  bucketStart: string;
  requests: number;
}

interface ActivityTimelineProps {
  presets: RangePreset[];
  // The preset defining the current extent (highlighted). Stays highlighted even
  // while a sub-window is selected, since it still describes how far back we span.
  extentKey: string;
  onPreset: (preset: RangePreset) => void;
  // A selected sub-window, resolved to absolute instants (end exclusive).
  onSelectRange: (startIso: string, endIso: string) => void;
  // The selection was widened back to the whole extent: fall back to the rolling
  // preset window rather than a bounded snapshot.
  onSelectFull: () => void;
  series: TimelinePoint[];
  bucket: UsageBucket;
  // The active window (server-echoed): positions the handles and drives the caption.
  windowStart?: string;
  windowEnd?: string;
  loading?: boolean;
  ariaLabel?: string;
  action?: ReactNode;
}

const BRAND = "var(--otari-brand)";
// Slider positions per bucket: fine enough to glide, coarse enough that arrow
// keys still move perceptibly (PageUp/Down jumps a whole bucket).
const STEPS_PER_BUCKET = 10;
// The pan strip's height: it overlays the x-axis band at the chart's bottom, so
// it never sits over the bars and their hover tooltips.
const PAN_STRIP_PX = 22;

function formatTick(iso: string, bucket: UsageBucket): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return bucket === "hour"
    ? d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", timeZone: "UTC" })
    : d.toLocaleDateString(undefined, { month: "short", day: "numeric", timeZone: "UTC" });
}

function TimelineTooltip({
  active,
  label,
  payload,
  bucket,
}: {
  active?: boolean;
  label?: string;
  payload?: readonly { value?: number | string }[];
  bucket: UsageBucket;
}) {
  const value = payload?.[0]?.value;
  if (!active || typeof value !== "number" || typeof label !== "string") return null;
  return (
    <div className="rounded-md border border-[var(--otari-line)] bg-[var(--otari-surface)] px-2 py-1 text-xs shadow-sm">
      <div className="text-[var(--otari-muted)]">{formatTick(label, bucket)} · UTC</div>
      <div className="font-medium tabular-nums text-[var(--otari-ink)]">{value.toLocaleString()} requests</div>
    </div>
  );
}

// A window-edge handle: a full-height grab bar with a centered grip pill, laid
// over the chart at the thumb's position. react-aria positions it with inline
// `left: X%` plus `transform: translate(-50%, -50%)`, so `top-1/2` is the whole
// centering recipe: adding Tailwind translate utilities (a separate `translate`
// property) would stack with that inline transform and shift the bar off the
// chart.
const THUMB_CLASS = "group top-1/2 h-full w-3 cursor-ew-resize outline-none";

// The thumbs' accessible names double as the routing key for the wrapper's
// arrow-key intercept below.
const THUMB_LABELS = ["Window start", "Window end"] as const;

function EdgeHandle({ index }: { index: 0 | 1 }) {
  return (
    <RacSliderThumb index={index} aria-label={THUMB_LABELS[index]} className={`${THUMB_CLASS} pointer-events-auto`}>
      <span aria-hidden className="absolute inset-y-0 left-1/2 w-0.5 -translate-x-1/2 bg-[var(--otari-brand)]" />
      <span
        aria-hidden
        className="absolute left-1/2 top-1/2 h-6 w-2 -translate-x-1/2 -translate-y-1/2 rounded-full border border-[var(--otari-brand)] bg-[var(--otari-surface)] shadow-sm group-focus-visible:ring-2 group-focus-visible:ring-[var(--otari-brand)]"
      />
    </RacSliderThumb>
  );
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
  const starts = series.map((p) => p.bucketStart);
  const data = series.map((p) => ({ x: p.bucketStart, requests: p.requests }));
  const n = series.length;
  const label = formatWindowLabel(windowStart, windowEnd);

  // Selection in fractional bucket units over [0, n]. Local state so a drag
  // updates every frame; synced from the active window only while not dragging
  // (the commit round-trips through the parent and must not fight the pointer).
  const windowToRange = (): [number, number] => {
    if (n === 0) return [0, 1];
    const { startIndex, endIndex } = bucketIndexRange(starts, windowStart, windowEnd);
    return [startIndex, endIndex + 1];
  };
  const [sel, setSelState] = useState<[number, number]>(windowToRange);
  // Mirror of `sel` for pointer handlers: a pointerup can fire before the last
  // pointermove's setState has re-rendered, and committing from the (stale)
  // closure value could snap to a different bucket than the one on screen.
  const selRef = useRef(sel);
  const setSel = (next: [number, number]) => {
    selRef.current = next;
    setSelState(next);
  };
  const dragging = useRef(false);
  useEffect(() => {
    if (!dragging.current) setSel(windowToRange());
    // starts is derived from series each render; resync on the window bounds and
    // on the extent itself (length plus its first/last bucket), not the fresh
    // array identity. The endpoints catch a series that rolls forward while the
    // bound strings and length stay identical, which would otherwise strand the
    // thumbs on stale positions.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [windowStart, windowEnd, n, starts[0], starts[n - 1]]);

  const commit = ([lo, hi]: [number, number]) => {
    dragging.current = false;
    if (n === 0) return;
    // Snap outward to whole buckets so the selection always covers what was
    // touched, and keep at least one bucket selected.
    const startIndex = Math.max(0, Math.min(n - 1, Math.floor(lo + 1e-6)));
    const endIndex = Math.max(startIndex, Math.min(n - 1, Math.ceil(hi - 1e-6) - 1));
    if (startIndex === 0 && endIndex === n - 1) {
      onSelectFull();
      setSel([0, n]);
      return;
    }
    const range = rangeFromBuckets(starts, startIndex, endIndex, bucket);
    if (range) onSelectRange(range.startIso, range.endIso);
  };

  // Step zoom. Out doubles the window around its center; at the full extent it
  // promotes to the next larger preset, so "the default window is too narrow"
  // is always one tap from a wider view. In halves it (min one bucket). When the
  // extent is not one of the presets (a drill-down window from another page),
  // fall back to the smallest preset that broadens it, so zoom-out never dead-ends.
  const extentIndex = presets.findIndex((p) => p.key === extentKey);
  const extentSeconds = extentIndex >= 0 ? presets[extentIndex].seconds : (n * bucketDurationMs(bucket)) / 1000;
  const largerPreset =
    extentIndex >= 0
      ? presets[extentIndex + 1]
      : presets.find((p) => p.seconds === null || (extentSeconds !== null && p.seconds > extentSeconds));
  const [lo, hi] = sel;
  const atFullExtent = lo <= 0.01 && hi >= n - 0.01;

  const applySpan = (newSpan: number) => {
    const span = Math.max(1, Math.min(n, newSpan));
    const center = (lo + hi) / 2;
    let nlo = center - span / 2;
    let nhi = center + span / 2;
    if (nlo < 0) {
      nhi -= nlo;
      nlo = 0;
    }
    if (nhi > n) {
      nlo -= nhi - n;
      nhi = n;
    }
    const next: [number, number] = [Math.max(0, nlo), Math.min(n, nhi)];
    setSel(next);
    commit(next);
  };

  const zoomOut = () => {
    if (atFullExtent) {
      if (largerPreset) onPreset(largerPreset);
      return;
    }
    applySpan((hi - lo) * 2);
  };
  const zoomIn = () => applySpan((hi - lo) / 2);

  // Arrow keys on a handle move that edge by one whole bucket. Intercepted in
  // the capture phase on the wrapper below (before react-aria's own handler,
  // which is unconditional): the slider's built-in arrow step is the fine
  // pointer step (0.1 bucket), which the bucket-snapping commit would undo,
  // making arrows a silent no-op. PageUp/Down, Home, and End stay react-aria's.
  const stepEdge = (index: number, deltaBuckets: number) => {
    const [slo, shi] = selRef.current;
    const next: [number, number] =
      index === 0
        ? [Math.max(0, Math.min(shi - 1, Math.round(slo) + deltaBuckets)), shi]
        : [slo, Math.min(n, Math.max(slo + 1, Math.round(shi) + deltaBuckets))];
    setSel(next);
    commit(next);
  };

  const onSliderKeyDownCapture = (event: React.KeyboardEvent) => {
    const delta =
      event.key === "ArrowRight" || event.key === "ArrowUp"
        ? 1
        : event.key === "ArrowLeft" || event.key === "ArrowDown"
          ? -1
          : 0;
    if (delta === 0) return;
    const which = (event.target as HTMLElement).getAttribute?.("aria-label");
    const index = THUMB_LABELS.indexOf(which as (typeof THUMB_LABELS)[number]);
    if (index < 0) return;
    event.preventDefault();
    event.stopPropagation();
    stepEdge(index, delta);
  };

  // Pan: dragging the axis strip slides the whole window. Plain pointer capture;
  // commits on release like the handles do.
  const areaRef = useRef<HTMLDivElement>(null);
  const pan = useRef<{ x: number; sel: [number, number] } | null>(null);
  const onPanMove = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!pan.current || !areaRef.current) return;
    const width = areaRef.current.getBoundingClientRect().width;
    if (width <= 0) return;
    const dx = ((event.clientX - pan.current.x) / width) * n;
    const span = pan.current.sel[1] - pan.current.sel[0];
    const plo = Math.max(0, Math.min(n - span, pan.current.sel[0] + dx));
    setSel([plo, plo + span]);
  };

  // Keyboard pan: slide the whole window by whole buckets without resizing it, so
  // a keyboard user can reach a mid-extent window with one control instead of
  // stepping both edges in turn. Commits like the pointer pan does; a clamped
  // pan (already against an edge) is a silent no-op. Mirrors the axis strip below.
  const panBy = (deltaBuckets: number) => {
    const [slo, shi] = selRef.current;
    const span = shi - slo;
    const plo = Math.max(0, Math.min(n - span, Math.round(slo) + deltaBuckets));
    const next: [number, number] = [plo, plo + span];
    setSel(next);
    commit(next);
  };

  const onPanKeyDown = (event: React.KeyboardEvent) => {
    if (atFullExtent) return;
    // Read the span from the live ref, the same source `panBy` reads its start
    // from: holding Page Up/Down repeats before the re-render lands, so deriving
    // the page size from render-state `hi - lo` would pan by a stale span.
    const [slo, shi] = selRef.current;
    const span = Math.max(1, Math.round(shi - slo));
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
                  : 0;
    if (delta === 0) return;
    event.preventDefault();
    panBy(delta);
  };

  const loPct = n ? (Math.min(lo, hi) / n) * 100 : 0;
  const hiPct = n ? (Math.max(lo, hi) / n) * 100 : 100;
  const zoomed = !atFullExtent;

  // Pan-strip ARIA. Its value is the window's *left edge*, which can only travel
  // up to `n - span`, so the reachable max is that, not `n`.
  const panSpan = Math.max(1, Math.round(hi - lo));
  const panMax = Math.max(0, n - panSpan);
  const panNow = Math.min(Math.max(0, Math.round(Math.min(lo, hi))), panMax);

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
          <span className="text-xs text-[var(--otari-muted)]">Showing {label} · UTC</span>
          {action}
        </div>
      </div>

      <div className="rounded-xl border border-[var(--otari-line)] bg-[var(--otari-surface)] p-2">
        <div className="flex items-center justify-between gap-2 px-1 pb-1">
          <span className="text-[11px] font-medium uppercase tracking-wide text-[var(--otari-muted)]">
            Requests / {bucket === "hour" ? "hour" : "day"}
          </span>
          <div className="flex items-center gap-1.5">
            <span className="hidden text-[11px] text-[var(--otari-muted)] sm:inline">
              drag the edges to zoom · the bottom strip to pan
            </span>
            <Button size="sm" variant="ghost" isIconOnly aria-label="Zoom in" isDisabled={n === 0} onPress={zoomIn}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="h-4 w-4" aria-hidden="true">
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
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="h-4 w-4" aria-hidden="true">
                <path d="M5 12h14" strokeLinecap="round" />
              </svg>
            </Button>
            {zoomed ? (
              <Button size="sm" variant="ghost" onPress={() => commit([0, n])}>
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
          <div className="flex h-[90px] items-center justify-center text-xs text-[var(--otari-muted)]">
            No activity in this range.
          </div>
        ) : (
          <div ref={areaRef} className="relative w-full">
            <div role="img" aria-label={ariaLabel} className="w-full touch-pan-y select-none">
              <ResponsiveContainer width="100%" height={90}>
                <BarChart data={data} margin={{ top: 4, right: 0, left: 0, bottom: 0 }}>
                  <XAxis
                    dataKey="x"
                    tickLine={false}
                    axisLine={false}
                    interval="preserveStartEnd"
                    minTickGap={40}
                    tickFormatter={(iso: string) => formatTick(iso, bucket)}
                    tick={{ fontSize: 10, fill: "var(--otari-muted)" }}
                  />
                  <Tooltip
                    cursor={{ fill: "var(--otari-line)", opacity: 0.35 }}
                    content={<TimelineTooltip bucket={bucket} />}
                  />
                  <Bar dataKey="requests" fill={BRAND} radius={[2, 2, 0, 0]} isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Out-of-window shades: pure CSS feedback that tracks the handles
                live without touching the recharts tree. pointer-events-none so
                the bars' hover tooltip keeps working. */}
            {zoomed ? (
              <>
                <div
                  aria-hidden
                  className="pointer-events-none absolute inset-y-0 left-0 bg-[var(--otari-bg)] opacity-70"
                  style={{ width: `${loPct}%` }}
                />
                <div
                  aria-hidden
                  className="pointer-events-none absolute inset-y-0 right-0 bg-[var(--otari-bg)] opacity-70"
                  style={{ width: `${100 - hiPct}%` }}
                />
              </>
            ) : null}

            {/* Pan strip along the axis band: slides the window without resizing
                it. Sits below the bars, so plot-area tooltips are unaffected. It
                is a single-value slider (the window's left edge, in buckets) so a
                keyboard user gets a real pan control; it stays in the tab order at
                the full extent but is announced disabled there, since a full-extent
                window has nothing to pan. */}
            <div
              role="slider"
              aria-label="Pan the selected window"
              aria-valuemin={0}
              aria-valuemax={panMax}
              aria-valuenow={panNow}
              aria-valuetext={`Window starting at ${formatTick(starts[Math.min(panNow, n - 1)] ?? starts[0], bucket)}`}
              aria-disabled={!zoomed}
              tabIndex={0}
              className="absolute bottom-0 z-[1] cursor-grab touch-none rounded bg-[var(--otari-brand)]/10 outline-none hover:bg-[var(--otari-brand)]/20 focus-visible:ring-2 focus-visible:ring-[var(--otari-brand)] active:cursor-grabbing"
              style={{ left: `${loPct}%`, width: `${Math.max(0, hiPct - loPct)}%`, height: PAN_STRIP_PX }}
              onKeyDown={onPanKeyDown}
              onPointerDown={(event) => {
                event.stopPropagation();
                event.preventDefault();
                dragging.current = true;
                pan.current = { x: event.clientX, sel: [Math.min(lo, hi), Math.max(lo, hi)] };
                event.currentTarget.setPointerCapture(event.pointerId);
              }}
              onPointerMove={onPanMove}
              onPointerUp={(event) => {
                event.currentTarget.releasePointerCapture(event.pointerId);
                pan.current = null;
                commit(selRef.current);
              }}
              onPointerCancel={(event) => {
                if (event.currentTarget.hasPointerCapture(event.pointerId)) {
                  event.currentTarget.releasePointerCapture(event.pointerId);
                }
                pan.current = null;
                commit(selRef.current);
              }}
            />

            {/* The selection handles, overlaid on the chart. The slider itself is
                pointer-transparent (so bars stay hoverable and a stray click
                cannot jump a handle); only the handles take the pointer. The
                wrapper's capture-phase keydown reroutes arrow keys to whole-
                bucket steps before react-aria's fine-step handler sees them. */}
            <div className="pointer-events-none absolute inset-0 z-[2]" onKeyDownCapture={onSliderKeyDownCapture}>
              <RacSlider
                aria-label="Selected time range"
                minValue={0}
                maxValue={n}
                step={1 / STEPS_PER_BUCKET}
                value={sel}
                onChange={(value) => {
                  dragging.current = true;
                  if (Array.isArray(value) && value.length === 2) setSel([value[0], value[1]]);
                }}
                onChangeEnd={(value) => {
                  if (Array.isArray(value) && value.length === 2) commit([value[0], value[1]]);
                }}
                className="pointer-events-none h-full w-full"
              >
                <RacSliderTrack className="pointer-events-none relative h-full w-full">
                  <EdgeHandle index={0} />
                  <EdgeHandle index={1} />
                </RacSliderTrack>
              </RacSlider>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
