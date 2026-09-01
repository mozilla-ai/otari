import type { ReactNode } from "react"
import { useRef, useState } from "react"
import {
  Bar,
  BarChart,
  Line,
  LineChart,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"

// Shared chart primitives for the dashboard, built on recharts. Pages compose
// these instead of hand-rolling SVG, so tooltips, responsive sizing, axis
// handling, and the drag-to-select interaction come from one place. Identity is
// never encoded by hue alone: every multi-series chart renders a legend, and
// the surrounding tables / captions stay the accessible source of truth.

const BRAND = "var(--color-primary)"

// Axis ticks are SVG `<text>`, so their size comes from a class rather than a
// `fontSize` number: `text-xs` is the type scale's 12px floor and nothing below
// it is legible. The fill stays a prop, which recharts writes as a presentation
// attribute.
const AXIS_TICK = {
  className: "text-xs",
  fill: "var(--color-text-muted)",
} as const

// One series of a (possibly stacked) trend chart. `color` is a CSS color,
// normally one of the fixed `--color-chart-cat-*` slots or a step of
// `--color-chart-ramp-*` (both validated per theme in globals.css), or a
// semantic token like `--color-danger`. Assign categorical slots in fixed
// order, never cycled.
export interface SeriesDef {
  key: string
  label: string
  color: string
}

// One x-axis bucket. `x` is the bucket identity (ISO instant for time series);
// series values live under their `SeriesDef.key`.
export type StackedPoint = { x: string } & Record<string, number | string>

// One point in a single-series trend: an x-axis/tooltip label and its value.
export interface ChartPoint {
  label: string
  value: number
}

// A series identity swatch. An SVG fill (which accepts var(--color-*) tokens)
// rather than an inline background-color style, per the dashboard's no-inline-
// styles convention.
function SeriesMarker({ color }: { color: string }) {
  return (
    <svg aria-hidden="true" viewBox="0 0 8 8" className="h-2 w-2 shrink-0">
      <rect width="8" height="8" rx="1.5" fill={color} />
    </svg>
  )
}

// Tooltip body. recharts clones this element and injects `active`, `payload`,
// and `label` at render time, so only the format props are passed by the
// caller. For a single series it shows one value row; for a stack it shows one
// row per non-zero series (marker + label + value) plus a total. Exported for
// direct branch testing since recharts hover is impractical to drive in jsdom.
export function ChartTooltip({
  active,
  label,
  payload,
  formatValue,
  formatLabel,
  showTotal = false,
}: {
  active?: boolean
  label?: ReactNode
  // recharts types a datum's value as `number | string`; guard before formatting.
  payload?: readonly {
    value?: number | string
    name?: ReactNode
    color?: string
  }[]
  formatValue: (value: number) => string
  formatLabel?: (label: string) => string
  showTotal?: boolean
}) {
  const rows = (payload ?? []).filter(
    (entry) => typeof entry.value === "number",
  )
  if (!active || rows.length === 0) {
    return null
  }
  const heading =
    typeof label === "string" && formatLabel ? formatLabel(label) : label
  // A stack hides its zero series (the row list stays scannable); a single
  // series keeps its zero row so hovering an empty bucket still reads a value.
  const visible =
    rows.length > 1 ? rows.filter((entry) => (entry.value as number) > 0) : rows
  const total = rows.reduce((sum, entry) => sum + (entry.value as number), 0)
  return (
    <div className="rounded-md border border-border bg-surface px-2.5 py-1.5 text-xs shadow-sm">
      <div className="text-muted">{heading}</div>
      <div className="mt-0.5 flex flex-col gap-0.5">
        {visible.map((entry, index) => (
          <div key={index} className="flex items-center justify-between gap-4">
            <span className="flex items-center gap-1.5 text-muted">
              {rows.length > 1 && entry.color ? (
                <SeriesMarker color={entry.color} />
              ) : null}
              {entry.name}
            </span>
            <span className="font-medium tabular-nums text-foreground">
              {formatValue(entry.value as number)}
            </span>
          </div>
        ))}
      </div>
      {showTotal && rows.length > 1 ? (
        <div className="mt-1 flex items-center justify-between gap-4 border-t border-border pt-1">
          <span className="text-muted">Total</span>
          <span className="font-semibold tabular-nums text-foreground">
            {formatValue(total)}
          </span>
        </div>
      ) : null}
    </div>
  )
}

// Legend chips for a multi-series chart: marker + label in text ink (identity
// rides the marker, never colored text). Single-series charts render no legend;
// their title names the series.
export function ChartLegend({ series }: { series: SeriesDef[] }) {
  if (series.length < 2) return null
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
      {series.map((s) => (
        <span
          key={s.key}
          className="flex items-center gap-1.5 text-xs text-muted"
        >
          <SeriesMarker color={s.color} />
          {s.label}
        </span>
      ))}
    </div>
  )
}

// The payload recharts hands to chart-level mouse handlers; only the active
// bucket index is consumed. It can arrive as a number or numeric string.
interface ChartMouseState {
  activeTooltipIndex?: number | string | null
}

function toIndex(state: ChartMouseState | null | undefined): number | null {
  const raw = state?.activeTooltipIndex
  const index = typeof raw === "string" ? Number(raw) : raw
  return typeof index === "number" && Number.isFinite(index) ? index : null
}

// A single-or-stacked bar trend over time with the industry-standard time
// selection: press and drag across the plot to select a bucket range (a live
// highlight tracks the pointer via recharts' own event coordinates, so axis
// margins never need pixel math), release to commit. A drag that never leaves
// its starting bucket is treated as a click and ignored, so hovering and
// clicking never zoom by accident. Releasing outside the plot commits the
// selection dragged so far.
//
// The y-axis is optional (the compact Activity strip omits it; the caller's
// caption carries the peak); stacked segments get a hairline surface stroke so
// adjacent fills always show their boundary. `window` dims buckets outside the
// active sub-window in place, which is how the Activity strip shows a zoomed
// selection inside its full extent.
export function TrendChart({
  data,
  series,
  formatValue,
  formatXTick,
  ariaLabel,
  height = 200,
  showYAxis = false,
  showTotal,
  onSelectRange,
  window: windowRange,
}: {
  data: StackedPoint[]
  series: SeriesDef[]
  formatValue: (value: number) => string
  formatXTick?: (x: string) => string
  ariaLabel: string
  height?: number
  showYAxis?: boolean
  showTotal?: boolean
  onSelectRange?: (startIndex: number, endIndex: number) => void
  window?: { startIndex: number; endIndex: number } | null
}) {
  const [drag, setDrag] = useState<{ start: number; end: number } | null>(null)
  // Mirror for the commit handlers: mouseup can fire before the last
  // mousemove's setState has re-rendered, and committing from the stale closure
  // would snap to the previous bucket.
  const dragRef = useRef(drag)
  const setDragBoth = (next: { start: number; end: number } | null) => {
    dragRef.current = next
    setDrag(next)
  }

  // Series indices arrive from tooltip state and props computed against
  // whatever length the data had at capture time; clamp every dereference so a
  // shrunken series degrades the highlight instead of blanking the chart.
  const last = data.length - 1
  const clampIndex = (index: number) =>
    Math.min(Math.max(index, 0), Math.max(last, 0))

  const commit = () => {
    const range = dragRef.current
    setDragBoth(null)
    if (!range || !onSelectRange || range.start === range.end) return
    // Clamp like the window prop below: the indices were captured from a
    // previous render's tooltip state, and a background refetch landing
    // mid-drag can shrink the series under them.
    const lo = clampIndex(Math.min(range.start, range.end))
    const hi = clampIndex(Math.max(range.start, range.end))
    if (lo === hi) return
    onSelectRange(lo, hi)
  }

  const selectable = Boolean(onSelectRange) && data.length > 1
  const dimmed =
    windowRange && data.length > 0
      ? {
          startIndex: clampIndex(windowRange.startIndex),
          endIndex: clampIndex(windowRange.endIndex),
        }
      : null
  const showDimming =
    dimmed !== null && (dimmed.startIndex > 0 || dimmed.endIndex < last)

  return (
    // A static chart is an image to AT; one that owns drag selection is not
    // (role="img" would hide the interaction entirely), so it presents as a
    // labeled group instead. Keyboard equivalents live with the callers
    // (presets, zoom buttons, the Activity pan rail).
    // biome-ignore lint/a11y/useAriaPropsSupportedByRole: both roles this switches between support aria-label; the rule cannot evaluate the condition
    <div
      role={selectable ? "group" : "img"}
      aria-label={ariaLabel}
      className={`w-full touch-pan-y select-none ${selectable ? "cursor-crosshair" : ""}`}
    >
      <ResponsiveContainer width="100%" height={height}>
        <BarChart
          data={data}
          margin={{ top: 4, right: 0, left: 0, bottom: 0 }}
          onMouseDown={(state) => {
            if (!selectable) return
            const index = toIndex(state)
            if (index !== null) setDragBoth({ start: index, end: index })
          }}
          onMouseMove={(state) => {
            const index = toIndex(state)
            if (dragRef.current && index !== null)
              setDragBoth({ ...dragRef.current, end: index })
          }}
          onMouseUp={commit}
          onMouseLeave={commit}
          onTouchStart={(state) => {
            if (!selectable) return
            const index = toIndex(state)
            if (index !== null) setDragBoth({ start: index, end: index })
          }}
          onTouchMove={(state) => {
            const index = toIndex(state)
            if (dragRef.current && index !== null)
              setDragBoth({ ...dragRef.current, end: index })
          }}
          onTouchEnd={commit}
        >
          <XAxis
            dataKey="x"
            tickLine={false}
            axisLine={false}
            interval="preserveStartEnd"
            minTickGap={40}
            tickFormatter={formatXTick}
            tick={AXIS_TICK}
          />
          {showYAxis ? (
            <YAxis
              width={52}
              tickLine={false}
              axisLine={false}
              tickFormatter={(value: number) => formatValue(value)}
              tick={AXIS_TICK}
            />
          ) : null}
          <Tooltip
            cursor={{ fill: "var(--color-border)", opacity: 0.35 }}
            content={
              <ChartTooltip
                formatValue={formatValue}
                formatLabel={formatXTick}
                showTotal={showTotal}
              />
            }
          />
          {series.map((s) => (
            <Bar
              key={s.key}
              dataKey={s.key}
              name={s.label}
              stackId="stack"
              fill={s.color}
              stroke="var(--color-surface)"
              strokeWidth={series.length > 1 ? 1 : 0}
              // Rounded data-ends only when nothing stacks on top; rounding
              // every stacked segment would fake gaps inside a column.
              radius={series.length > 1 ? 0 : [2, 2, 0, 0]}
              isAnimationActive={false}
            />
          ))}
          {/* Out-of-window dimming (start side, then end side), under the drag
              highlight. ReferenceArea x-bounds are category values, so the
              shading stays glued to its buckets across resizes. */}
          {showDimming && dimmed && dimmed.startIndex > 0 ? (
            <ReferenceArea
              x1={data[0].x}
              x2={data[dimmed.startIndex - 1].x}
              fill="var(--color-surface-muted)"
              fillOpacity={0.75}
              stroke="none"
            />
          ) : null}
          {showDimming && dimmed && dimmed.endIndex < last ? (
            <ReferenceArea
              x1={data[dimmed.endIndex + 1].x}
              x2={data[data.length - 1].x}
              fill="var(--color-surface-muted)"
              fillOpacity={0.75}
              stroke="none"
            />
          ) : null}
          {drag && clampIndex(drag.start) !== clampIndex(drag.end) ? (
            <ReferenceArea
              x1={data[clampIndex(Math.min(drag.start, drag.end))].x}
              x2={data[clampIndex(Math.max(drag.start, drag.end))].x}
              fill={BRAND}
              fillOpacity={0.18}
              stroke={BRAND}
              strokeOpacity={0.5}
            />
          ) : null}
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

// A compact, axis-free trend line for KPI tiles. Conveys shape only: no ticks,
// no tooltip, one color. `ariaLabel` should describe what the trend is (e.g.
// "Spend trend over the selected window") so it is legible without the visual.
export function Sparkline({
  values,
  ariaLabel,
  height = 32,
}: {
  values: number[]
  ariaLabel: string
  height?: number
}) {
  const data = values.map((value, index) => ({ index, value }))
  return (
    <div role="img" aria-label={ariaLabel} className="w-full">
      <ResponsiveContainer width="100%" height={height}>
        <LineChart
          data={data}
          margin={{ top: 2, right: 2, left: 2, bottom: 2 }}
        >
          <Line
            type="monotone"
            dataKey="value"
            stroke={BRAND}
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
