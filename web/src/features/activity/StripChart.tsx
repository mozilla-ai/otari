import { Spinner } from "@heroui/react"
import { useEffect, useRef, useState } from "react"
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  useActiveTooltipLabel,
  useIsTooltipActive,
  usePlotArea,
  XAxis,
  YAxis,
} from "recharts"
import {
  type ChartDrag,
  releasedSelection,
} from "@/design-system/metrics/charts"
import { formatNumber } from "@/shared/helpers/format"
import {
  type Bucket,
  barBox,
  barCategoryGap,
  minPointSize,
  xLabelWidth,
  xTickIndices,
  yTicks,
} from "./chartMath"

// The timeline's bar drawing, built on Recharts from our own tokens. The
// interaction (press and drag to select, click a bucket, touch) is
// `TrendChart`'s, so it filters the table the same way; the failures sit in a
// lane under the baseline, and the numbers live in the header, which reads out
// what the pointer is over, so there is no tooltip.

interface StripPoint extends Bucket {
  x: string
}

export interface Pointed {
  lo: number
  hi: number
}

interface Geometry {
  top: number
  plot: number
  // Below the plot: the baseline, a 2px gap and the failure lane.
  below: number
  gutter: number
  axis: number
  height: number
}

const LABEL_ROW = 20

const PLOT_HEIGHT = 72
const BELOW_PLOT = 1 + 2 + 6

function geometry(phone: boolean): Geometry {
  const top = phone ? 8 : 6
  return {
    top,
    plot: PLOT_HEIGHT,
    below: BELOW_PLOT,
    gutter: phone ? 32 : 40,
    axis: BELOW_PLOT + LABEL_ROW,
    height: top + PLOT_HEIGHT + BELOW_PLOT + LABEL_ROW,
  }
}

const OUTSIDE_OPACITY = 0.35
const FADE =
  "transition-[fill-opacity] duration-100 ease-out motion-reduce:transition-none"

interface ChartMouseState {
  activeTooltipIndex?: number | string | null
}

function toIndex(state: ChartMouseState | null | undefined): number | null {
  const raw = state?.activeTooltipIndex
  const index = typeof raw === "string" ? Number(raw) : raw
  return typeof index === "number" && Number.isFinite(index) ? index : null
}

const TOUCH_ECHO_MS = 1000

// Where a bucket's slot and bar sit inside the plot.
function slotOf(
  index: number,
  plot: { x: number; width: number },
  count: number,
  gap: `${number}%` | 0,
) {
  const slot = plot.width / count
  const { offset, size } = barBox(slot, gap)
  const slotX = plot.x + index * slot
  return { slotX, slot, barX: slotX + offset, bar: size }
}

function XTick({
  x,
  value,
  last,
  text,
  baselineY,
}: {
  x: number
  value: string
  last: string
  text: string
  baselineY: (plotBottom: number) => number
}) {
  const plot = usePlotArea()
  if (!plot) return null
  const isLast = value === last
  return (
    <text
      x={isLast ? plot.x + plot.width : x}
      y={baselineY(plot.y + plot.height)}
      textAnchor={isLast ? "end" : "middle"}
      fill="var(--color-text-muted)"
      className="text-xs"
    >
      {text}
    </text>
  )
}

// The bucket recharts highlights, whether the pointer or the arrow keys put it
// there: the keys move it with no mouse event, so the header reads it here.
function ActiveBucket({
  data,
  onChange,
}: {
  data: StripPoint[]
  onChange: (index: number | null) => void
}) {
  const isActive = useIsTooltipActive()
  const label = useActiveTooltipLabel()
  const index = isActive ? data.findIndex((point) => point.x === label) : -1
  const onChangeRef = useRef(onChange)
  onChangeRef.current = onChange
  useEffect(() => {
    onChangeRef.current(index < 0 ? null : index)
  }, [index])
  return null
}

function Layer({
  data,
  window: win,
  drag,
  gap,
  geo,
  empty,
}: {
  data: StripPoint[]
  window?: { startIndex: number; endIndex: number }
  drag?: Pointed
  gap: `${number}%` | 0
  geo: Geometry
  empty: boolean
}) {
  const plot = usePlotArea()
  if (!plot) return null
  const bottom = plot.y + plot.height
  const n = data.length
  const outside = (index: number) =>
    win !== undefined && (index < win.startIndex || index > win.endIndex)
  return (
    <g>
      {drag && n > 0 ? (
        <>
          {/* The part of the selection the plot's own ReferenceArea cannot reach. */}
          <rect
            x={slotOf(drag.lo, plot, n, gap).slotX}
            y={bottom}
            width={
              slotOf(drag.hi, plot, n, gap).slotX +
              plot.width / n -
              slotOf(drag.lo, plot, n, gap).slotX
            }
            height={geo.below}
            fill="var(--color-primary-subtle)"
          />
          <line
            x1={slotOf(drag.lo, plot, n, gap).slotX + 0.5}
            x2={slotOf(drag.lo, plot, n, gap).slotX + 0.5}
            y1={plot.y}
            y2={bottom + geo.below}
            stroke="var(--color-primary)"
          />
          <line
            x1={slotOf(drag.hi, plot, n, gap).slotX + plot.width / n - 0.5}
            x2={slotOf(drag.hi, plot, n, gap).slotX + plot.width / n - 0.5}
            y1={plot.y}
            y2={bottom + geo.below}
            stroke="var(--color-primary)"
          />
        </>
      ) : null}
      <line
        x1={plot.x}
        x2={plot.x + plot.width}
        y1={bottom + 0.5}
        y2={bottom + 0.5}
        stroke="var(--color-border-strong)"
      />
      {!empty
        ? data.map((point, index) => {
            if (point.errors <= 0) return null
            const { barX, bar } = slotOf(index, plot, n, gap)
            return (
              <rect
                key={point.x}
                x={barX}
                y={bottom + 3}
                width={bar}
                height={6}
                fill="var(--color-danger)"
                fillOpacity={outside(index) ? OUTSIDE_OPACITY : 1}
                className={FADE}
              />
            )
          })
        : null}
    </g>
  )
}

export function StripChart({
  data,
  formatTick,
  ariaLabel,
  window: windowRange,
  onSelectRange,
  onReading,
  phone,
  loading,
  emptyLabel,
}: {
  data: StripPoint[]
  formatTick: (iso: string) => string
  ariaLabel: string
  window?: { startIndex: number; endIndex: number }
  onSelectRange: (startIndex: number, endIndex: number) => void
  // What the pointer is over: a bucket, or the range being dragged. Null when nothing.
  onReading: (over: Pointed | null) => void
  phone: boolean
  loading: boolean
  emptyLabel: string
}) {
  const geo = geometry(phone)
  const n = data.length
  const [width, setWidth] = useState(0)
  const [hover, setHover] = useState<number | null>(null)
  const [drag, setDrag] = useState<ChartDrag>()
  const dragRef = useRef(drag)
  const setDragBoth = (next?: ChartDrag) => {
    dragRef.current = next
    setDrag(next)
  }
  const lastTouchRef = useRef(0)
  const last = n - 1
  const clamp = (index: number) =>
    Math.min(Math.max(index, 0), Math.max(last, 0))

  const plotWidth = Math.max(0, width - geo.gutter)
  const pitch = n > 0 && plotWidth > 0 ? plotWidth / n : 35
  const gap = barCategoryGap(pitch)

  const max = data.reduce((m, point) => Math.max(m, point.requests), 0)
  const empty = n === 0 || max === 0
  // Three ticks, the 0 label dropped: this plot is a shape to drag on, and the
  // header carries the numbers.
  const scale = yTicks(max, 3)
  const yShown = empty ? [] : scale.ticks.filter((tick) => tick > 0)
  const xs = xTickIndices(
    n,
    pitch,
    phone ? 64 : 72,
    xLabelWidth(data.map((point) => formatTick(point.x))),
    geo.gutter,
  ).map((index) => data[index].x)
  const lastX = data[last]?.x ?? ""

  const dragged = drag
    ? {
        lo: clamp(Math.min(drag.start, drag.end)),
        hi: clamp(Math.max(drag.start, drag.end)),
      }
    : undefined
  const over: Pointed | null =
    dragged ?? (hover !== null ? { lo: clamp(hover), hi: clamp(hover) } : null)
  const overKey = over ? `${over.lo}:${over.hi}` : ""
  const onReadingRef = useRef(onReading)
  onReadingRef.current = onReading
  // biome-ignore lint/correctness/useExhaustiveDependencies: keyed on the range's value, not its identity
  useEffect(() => {
    onReadingRef.current(overKey ? over : null)
  }, [overKey])

  const cells = data.map((point, index) => (
    <Cell
      key={point.x}
      fillOpacity={
        windowRange &&
        (index < windowRange.startIndex || index > windowRange.endIndex)
          ? OUTSIDE_OPACITY
          : 1
      }
    />
  ))

  const isSelectable = n > 1
  const begin = (index: number | null) => {
    if (index !== null) setDragBoth({ start: index, end: index, moved: false })
  }
  const extend = (index: number | null) => {
    const current = dragRef.current
    if (current && index !== null) {
      // Same bucket as the last move: nothing to redraw.
      if (index === current.end) return
      setDragBoth({
        start: current.start,
        end: index,
        moved: current.moved || index !== current.start,
      })
    }
  }
  const commit = (how: "release" | "leave" | "touch") => {
    const picked = releasedSelection(dragRef.current, how, last)
    setDragBoth(undefined)
    if (picked.length === 2) onSelectRange(picked[0], picked[1])
  }
  const isTouchEcho = () => Date.now() - lastTouchRef.current < TOUCH_ECHO_MS

  return (
    // biome-ignore lint/a11y/useAriaPropsSupportedByRole: both roles this switches between support aria-label; the rule cannot evaluate the condition
    <div
      role={isSelectable ? "group" : "img"}
      aria-label={ariaLabel}
      className={`relative w-full touch-pan-y select-none [&_:focus:not(:focus-visible)]:outline-none [&_.recharts-bar-rectangle_path]:transition-[fill-opacity] [&_.recharts-bar-rectangle_path]:duration-100 [&_.recharts-bar-rectangle_path]:ease-out motion-reduce:[&_.recharts-bar-rectangle_path]:transition-none ${isSelectable ? "cursor-crosshair" : ""}`}
      onTouchCancel={() => setDragBoth(undefined)}
    >
      <ResponsiveContainer
        width="100%"
        height={geo.height}
        onResize={(w) => setWidth(w)}
      >
        <BarChart
          data={data}
          margin={{ top: geo.top, right: 0, left: 0, bottom: 0 }}
          barCategoryGap={gap}
          onMouseDown={(state) => {
            if (isSelectable && !isTouchEcho()) begin(toIndex(state))
          }}
          onMouseMove={(state) => extend(toIndex(state))}
          onMouseUp={() => commit("release")}
          onMouseLeave={() => commit("leave")}
          onTouchStart={(state) => {
            lastTouchRef.current = Date.now()
            if (isSelectable) begin(toIndex(state))
          }}
          onTouchMove={(state) => {
            lastTouchRef.current = Date.now()
            extend(toIndex(state))
          }}
          onTouchEnd={() => {
            lastTouchRef.current = Date.now()
            commit("touch")
          }}
        >
          <CartesianGrid
            vertical={false}
            stroke="var(--color-border)"
            horizontalPoints={yShown.map(
              (tick) => geo.top + geo.plot * (1 - tick / scale.top) + 0.5,
            )}
          />
          <XAxis
            dataKey="x"
            ticks={xs}
            interval={0}
            tickLine={false}
            axisLine={false}
            height={geo.axis}
            tick={(props) => (
              <XTick
                x={Number(props.x)}
                value={String(props.payload.value)}
                last={lastX}
                text={formatTick(String(props.payload.value))}
                baselineY={(bottom) => bottom + geo.below + 16}
              />
            )}
          />
          <YAxis
            interval={0}
            width={geo.gutter}
            domain={[0, empty ? 1 : scale.top]}
            ticks={yShown}
            tickLine={false}
            axisLine={false}
            tickFormatter={(value: number) => formatNumber(value)}
            tick={{ className: "text-xs", fill: "var(--color-text-muted)" }}
          />
          <Tooltip
            isAnimationActive={false}
            cursor={
              dragged && dragged.lo !== dragged.hi
                ? false
                : { fill: "var(--color-surface-subtle)", fillOpacity: 0.6 }
            }
            content={() => null}
          />
          {/* Recharts keeps a touched bucket highlighted after the finger
              lifts, which is not a hover. */}
          <ActiveBucket
            data={data}
            onChange={(index) => setHover(isTouchEcho() ? null : index)}
          />
          <Bar
            dataKey="requests"
            fill="var(--color-primary)"
            maxBarSize={40}
            radius={0}
            minPointSize={minPointSize}
            isAnimationActive={false}
          >
            {cells}
          </Bar>
          {dragged && dragged.lo !== dragged.hi ? (
            <ReferenceArea
              x1={data[dragged.lo].x}
              x2={data[dragged.hi].x}
              fill="var(--color-primary-subtle)"
              fillOpacity={1}
              stroke="none"
            />
          ) : null}
          <Layer
            data={data}
            window={windowRange}
            drag={dragged && dragged.lo !== dragged.hi ? dragged : undefined}
            gap={gap}
            geo={geo}
            empty={empty}
          />
        </BarChart>
      </ResponsiveContainer>
      {empty ? (
        <div
          className="pointer-events-none absolute inset-x-0 flex items-center justify-center text-sm text-muted"
          style={{ top: geo.top, height: geo.plot }}
        >
          {loading ? <Spinner size="sm" /> : emptyLabel}
        </div>
      ) : null}
    </div>
  )
}
