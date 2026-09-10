import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"
import { RefreshButton } from "@/design-system/actions/RefreshButton"
import { ACTIVITY_PRESETS } from "@/shared/helpers/timeRange"
import type { TimelinePoint } from "./ActivityTimeline"
import { ActivityTimeline } from "./ActivityTimeline"

/**
 * The Activity page's context strip: a request-volume histogram that doubles as
 * the time-range selector.
 *
 * Presets set the *extent* the strip spans; **dragging across the chart selects
 * the window**, and regions outside it dim in place. Failed requests render as a
 * segment stacked on top of each bar, so dropped traffic is visible on the strip
 * itself rather than only in the table below.
 *
 * Buckets are UTC-aligned so the selection lines up with the bars, which is why
 * every fixture below is built on UTC instants.
 */
const HOURS = Array.from({ length: 24 }, (_, index) =>
  new Date(Date.UTC(2026, 7, 24, index)).toISOString(),
)

const QUIET: TimelinePoint[] = HOURS.map((bucketStart, index) => ({
  bucketStart,
  requests: 180 + Math.round(Math.sin(index / 3) * 90),
}))

const WITH_ERRORS: TimelinePoint[] = HOURS.map((bucketStart, index) => ({
  bucketStart,
  requests: 180 + Math.round(Math.sin(index / 3) * 90),
  // A burst in the small hours, which is the shape an incident actually has.
  errors:
    index >= 9 && index <= 13 ? 40 + (index - 9) * 22 : index % 7 === 0 ? 4 : 0,
}))

const meta = {
  title: "Dashboard/Activity/ActivityTimeline",
  component: ActivityTimeline,
  args: {
    presets: ACTIVITY_PRESETS,
    extentKey: "24h",
    onPreset: () => {},
    onSelectRange: () => {},
    onSelectFull: () => {},
    series: QUIET,
    bucket: "hour",
  },
  parameters: { layout: "padded" },
} satisfies Meta<typeof ActivityTimeline>

export default meta

type Story = StoryObj<typeof meta>

/** No failures in the window, so no error series is stacked at all. */
export const NoErrors: Story = {
  render: (args) => (
    <div className="w-[52rem]">
      <ActivityTimeline {...args} />
    </div>
  ),
}

/** With failures, which adds the red segment and the two-series legend. */
export const WithErrors: Story = {
  args: { series: WITH_ERRORS },
  render: (args) => (
    <div className="w-[52rem]">
      <ActivityTimeline {...args} />
    </div>
  ),
}

/**
 * A window narrower than the extent. The bars outside it dim in place rather
 * than being dropped, so the selection stays legible against the whole span.
 */
export const ZoomedWindow: Story = {
  args: {
    series: WITH_ERRORS,
    windowStart: HOURS[8],
    windowEnd: HOURS[15],
  },
  render: (args) => (
    <div className="w-[52rem]">
      <ActivityTimeline {...args} />
    </div>
  ),
}

export const Loading: Story = {
  args: { series: [], loading: true },
  render: (args) => (
    <div className="w-[52rem]">
      <ActivityTimeline {...args} />
    </div>
  ),
}

/** A gateway that has served nothing yet. */
export const NoTraffic: Story = {
  args: { series: HOURS.map((bucketStart) => ({ bucketStart, requests: 0 })) },
  render: (args) => (
    <div className="w-[52rem]">
      <ActivityTimeline {...args} />
    </div>
  ),
}

/** Daily buckets over a month, which is what the 30d preset selects. */
export const DailyBuckets: Story = {
  args: {
    extentKey: "30d",
    bucket: "day",
    series: Array.from({ length: 30 }, (_, index) => ({
      bucketStart: new Date(Date.UTC(2026, 6, 26 + index)).toISOString(),
      requests: 3_200 + Math.round(Math.cos(index / 4) * 1_400) + index * 60,
      errors: index % 6 === 0 ? 210 : 24,
    })),
  },
  render: (args) => (
    <div className="w-[52rem]">
      <ActivityTimeline {...args} />
    </div>
  ),
}

/** The `action` slot, which the page uses for its refresh control. */
export const WithAction: Story = {
  args: {
    series: WITH_ERRORS,
    action: (
      <RefreshButton onRefresh={() => {}} updatedAt={Date.now() - 30_000} />
    ),
  },
  render: (args) => (
    <div className="w-[52rem]">
      <ActivityTimeline {...args} />
    </div>
  ),
}

/**
 * Wired up, so the presets and the drag-to-select actually move the window.
 * Drag across the bars.
 */
export const Interactive: Story = {
  render: (args) => {
    const [extentKey, setExtentKey] = useState("24h")
    const [range, setRange] = useState<{ start?: string; end?: string }>({})
    return (
      <div className="flex w-[52rem] flex-col gap-2">
        <ActivityTimeline
          {...args}
          series={WITH_ERRORS}
          extentKey={extentKey}
          windowStart={range.start}
          windowEnd={range.end}
          onPreset={(preset) => {
            setExtentKey(preset.key)
            setRange({})
          }}
          onSelectRange={(start, end) => setRange({ start, end })}
          onSelectFull={() => setRange({})}
        />
        <p className="text-caption">
          {range.start
            ? `Window: ${range.start} to ${range.end}`
            : `Extent: ${extentKey}, no sub-window selected.`}
        </p>
      </div>
    )
  },
}
