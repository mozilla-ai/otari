import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"

import {
  ChartLegend,
  ChartTooltip,
  type SeriesDef,
  Sparkline,
  type StackedPoint,
  TrendChart,
} from "./charts"

/**
 * The chart pieces, in one file because `charts.tsx` exports them as a set and
 * choosing between them is one decision.
 *
 * Series colors are consumed as `var(--color-chart-cat-N)`, not as a Tailwind
 * class: the chart tokens are deliberately not registered in `@theme`, because
 * recharts needs a resolvable CSS value rather than a utility. Assign them in
 * order and never cycle, a ninth series is `--color-chart-cat-other`, not
 * `cat-1` again.
 *
 * Every chart needs a **width-bearing parent**: recharts' `ResponsiveContainer`
 * measures it, so a chart inside an unsized box collapses to nothing.
 */
// Required props on the meta, so each story can supply its own `render` (every
// chart here needs a sized wrapper) without restating the component's contract.
const meta = {
  title: "Design system/Metrics/Charts",
  component: TrendChart,
  args: {
    data: [],
    series: [],
    formatValue: (value: number) => String(value),
    ariaLabel: "Trend",
  },
  parameters: { layout: "padded" },
} satisfies Meta<typeof TrendChart>

export default meta

type Story = StoryObj<typeof meta>

const DAYS = Array.from({ length: 14 }, (_, index) => {
  const day = new Date(Date.UTC(2026, 7, 1 + index))
  return day.toISOString()
})

const SINGLE: StackedPoint[] = DAYS.map((x, index) => ({
  x,
  requests: 900 + Math.round(Math.sin(index / 2) * 400) + index * 70,
}))

const SINGLE_SERIES: SeriesDef[] = [
  { key: "requests", label: "Requests", color: "var(--color-chart-cat-1)" },
]

const STACK: StackedPoint[] = DAYS.map((x, index) => ({
  x,
  succeeded: 800 + Math.round(Math.cos(index / 3) * 220) + index * 40,
  refused: index % 5 === 0 ? 140 : 30,
  failed: index % 4 === 0 ? 90 : 12,
}))

const STACK_SERIES: SeriesDef[] = [
  { key: "succeeded", label: "Succeeded", color: "var(--color-chart-cat-1)" },
  { key: "refused", label: "Refused", color: "var(--color-chart-cat-2)" },
  { key: "failed", label: "Failed", color: "var(--color-chart-cat-3)" },
]

const count = (value: number) => value.toLocaleString("en-US")
const day = (x: string) =>
  new Date(x).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  })

/** One series. No legend is rendered: its title names the series. */
export const TrendSingleSeries: Story = {
  render: () => (
    <div className="w-[48rem]">
      <TrendChart
        data={SINGLE}
        series={SINGLE_SERIES}
        formatValue={count}
        formatXTick={day}
        ariaLabel="Requests per day"
      />
    </div>
  ),
}

/** Stacked, with the legend and a tooltip total. Hover a bucket. */
export const TrendStacked: Story = {
  render: () => (
    <div className="flex w-[48rem] flex-col gap-2">
      <ChartLegend series={STACK_SERIES} />
      <TrendChart
        data={STACK}
        series={STACK_SERIES}
        formatValue={count}
        formatXTick={day}
        ariaLabel="Requests per day by outcome"
        showTotal
        showYAxis
      />
    </div>
  ),
}

/**
 * Drag across the plot to select a range. The chart reports bucket indices and
 * the caller decides what that means; here it just narrows `window`, which is how
 * the chart renders a selection back.
 */
export const TrendWithRangeSelection: Story = {
  render: () => {
    const [range, setRange] = useState<{
      startIndex: number
      endIndex: number
    } | null>({ startIndex: 3, endIndex: 8 })
    return (
      <div className="flex w-[48rem] flex-col gap-2">
        <TrendChart
          data={SINGLE}
          series={SINGLE_SERIES}
          formatValue={count}
          formatXTick={day}
          ariaLabel="Requests per day"
          window={range}
          onSelectRange={(startIndex, endIndex) =>
            setRange({ startIndex, endIndex })
          }
        />
        <p className="text-caption">
          {range
            ? `Buckets ${range.startIndex}-${range.endIndex} selected. Drag to change.`
            : "Drag across the plot to select a range."}
        </p>
      </div>
    )
  },
}

/** A single empty bucket set, which is what a fresh gateway has. */
export const TrendEmpty: Story = {
  render: () => (
    <div className="w-[48rem]">
      <TrendChart
        data={DAYS.map((x) => ({ x, requests: 0 }))}
        series={SINGLE_SERIES}
        formatValue={count}
        formatXTick={day}
        ariaLabel="Requests per day"
      />
    </div>
  ),
}

/**
 * Axis-free, for a KPI tile. It takes bare numbers rather than a series, because
 * at 32px high there is nothing to label.
 */
export const Sparklines: Story = {
  render: () => (
    <div className="flex w-[36rem] flex-col gap-4">
      {[
        { label: "climbing", values: [12, 18, 15, 24, 22, 31, 28, 36, 34, 41] },
        { label: "flat", values: [22, 21, 23, 22, 22, 23, 21, 22, 23, 22] },
        { label: "falling", values: [41, 38, 34, 36, 29, 27, 22, 19, 15, 11] },
        { label: "spiky", values: [4, 6, 5, 38, 7, 5, 6, 41, 8, 6] },
      ].map((row) => (
        <div key={row.label} className="flex items-center gap-4">
          <span className="w-[6rem] shrink-0 text-caption">{row.label}</span>
          <div className="flex-1">
            <Sparkline values={row.values} ariaLabel={`${row.label} trend`} />
          </div>
        </div>
      ))}
    </div>
  ),
}

/**
 * The legend on its own. Under two series it renders nothing, because a
 * single-series chart's title already names it.
 */
export const Legend: Story = {
  render: () => (
    <div className="flex flex-col gap-3">
      <ChartLegend series={STACK_SERIES} />
      <ChartLegend series={SINGLE_SERIES} />
      <p className="text-caption">
        Nothing rendered for the single series above.
      </p>
    </div>
  ),
}

/**
 * The tooltip body, rendered directly.
 *
 * recharts clones it and injects `active`, `payload` and `label` at hover time,
 * which is impractical to drive in jsdom, so it is exported for direct testing,
 * and here for direct viewing. A stack hides its zero series to stay scannable; a
 * single series keeps its zero row, so hovering an empty bucket still reads a value.
 */
export const Tooltips: Story = {
  render: () => (
    <div className="flex flex-wrap items-start gap-6">
      <ChartTooltip
        active
        label={DAYS[6]}
        formatLabel={day}
        formatValue={count}
        payload={[{ value: 1284, name: "Requests" }]}
      />
      <ChartTooltip
        active
        label={DAYS[6]}
        formatLabel={day}
        formatValue={count}
        showTotal
        payload={[
          { value: 1284, name: "Succeeded", color: "var(--color-chart-cat-1)" },
          { value: 140, name: "Refused", color: "var(--color-chart-cat-2)" },
          { value: 0, name: "Failed", color: "var(--color-chart-cat-3)" },
        ]}
      />
      <ChartTooltip
        active
        label={DAYS[2]}
        formatLabel={day}
        formatValue={count}
        payload={[{ value: 0, name: "Requests" }]}
      />
    </div>
  ),
}

/**
 * `xTickInterval` and `yTickCount` set the axis density, and both exist because
 * recharts' defaults are wrong for a strip this shape.
 *
 * The x default is `"preserveStartEnd"`, which thins labels by pixel gap, so the
 * density becomes a function of the window width rather than of the data: the
 * same chart shows every other day on a wide screen and every fifth on a narrow
 * one. A number pins it to the data instead. The y default is 5, which is more
 * gridlines than a short strip earns.
 *
 * Compare the rows: same data, three densities.
 */
export const AxisDensity: Story = {
  render: () => (
    <div className="flex w-[48rem] flex-col gap-6">
      {(
        [
          ["recharts' defaults", undefined, undefined],
          ["xTickInterval 1, yTickCount 3", 1, 3],
          ["xTickInterval 6, yTickCount 2", 6, 2],
        ] as const
      ).map(([label, xTickInterval, yTickCount]) => (
        <div key={label} className="flex flex-col gap-1">
          <span className="text-overline">{label}</span>
          <TrendChart
            data={STACK}
            series={STACK_SERIES}
            formatValue={count}
            formatXTick={day}
            ariaLabel={`Requests per day, ${label}`}
            showYAxis
            xTickInterval={xTickInterval}
            yTickCount={yTickCount}
          />
        </div>
      ))}
    </div>
  ),
}
