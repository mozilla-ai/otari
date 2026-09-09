import type { Meta, StoryObj } from "@storybook/react-vite"
import { KpiCell } from "./KpiCell"
import { KpiStrip } from "./KpiStrip"
import { Meter } from "./Meter"
import { TrendChip } from "./TrendChip"

/**
 * The four figures a page opens with, and the cells inside it.
 *
 * Documented as a pair because neither is useful alone: the strip owns the grid
 * and the empty state, the cell owns one figure. Replaces
 * `deprecated/StatCard`, which put each figure in a card and is still on four
 * uses in Overview.
 *
 * The figure is `text-mono-figure`: 30px, mono, weight 400. It is larger than
 * the 28px page title and still does not fight it, because the title is display
 * at 550. That is the ladder rule, and it is why a figure never goes semibold:
 * at 30px the size is already the hierarchy.
 */
const meta = {
  title: "Design system/Metrics/KpiStrip",
  component: KpiStrip,
  args: { empty: false, children: null },
  parameters: { layout: "padded" },
} satisfies Meta<typeof KpiStrip>

export default meta

type Story = StoryObj<typeof meta>

export const Default: Story = {
  render: () => (
    <KpiStrip empty={false}>
      <KpiCell label="Spend" value="$412.08" subline="This month" />
      <KpiCell label="Requests" value="1,284,901" subline="This month" />
      <KpiCell label="Tokens" value="94.2M" subline="This month" />
      <KpiCell label="Errors" value="0.4%" subline="This month" />
    </KpiStrip>
  ),
}

/**
 * `delta` slots a `TrendChip` under the figure, and `graphic` slots a meter or
 * a sparkline. Both are nodes rather than data, so the cell never has to know
 * what a trend or a bar is.
 */
export const WithDeltaAndGraphic: Story = {
  render: () => (
    <KpiStrip empty={false}>
      <KpiCell
        label="Spend"
        value="$412.08"
        subline="vs last month"
        delta={<TrendChip fraction={0.184} polarity="down-is-good" />}
      />
      <KpiCell
        label="Budget used"
        value="41%"
        subline="$412 of $1,000"
        graphic={<Meter fraction={0.41} ariaLabel="41% of the budget used" />}
      />
      <KpiCell
        label="Requests"
        value="1,284,901"
        subline="vs last month"
        delta={<TrendChip fraction={-0.021} polarity="up-is-good" />}
      />
      <KpiCell
        label="Errors"
        value="0.4%"
        subline="vs last month"
        delta={<TrendChip fraction={0.003} polarity="down-is-good" />}
      />
    </KpiStrip>
  ),
}

/**
 * `severity` puts a `SeverityMark` beside the figure: a word next to a mark,
 * never a color alone.
 */
export const WithSeverity: Story = {
  render: () => (
    <KpiStrip empty={false}>
      <KpiCell
        label="Budget"
        value="$1,041.20"
        severity={{ status: "alert", word: "over" }}
        subline="$1,000 allocated"
      />
      <KpiCell
        label="Providers"
        value="3 of 4"
        severity={{ status: "warn", word: "degraded" }}
        subline="Anthropic is retrying"
      />
      <KpiCell
        label="Keys"
        value="12"
        severity={{ status: "ok", word: "active" }}
        subline="None revoked"
      />
      <KpiCell label="Workspaces" value="3" subline="All within budget" />
    </KpiStrip>
  ),
}

/**
 * `empty` for a deployment that has served nothing yet. Every cell still says
 * what it is and why it is empty: an em dash with nothing under it makes the
 * reader guess whether the number is missing or zero.
 */
export const Empty: Story = {
  render: () => (
    <KpiStrip empty>
      <KpiCell label="Spend" value="—" subline="No requests yet" />
      <KpiCell label="Requests" value="—" subline="No requests yet" />
      <KpiCell label="Tokens" value="—" subline="No requests yet" />
      <KpiCell label="Errors" value="—" subline="No requests yet" />
    </KpiStrip>
  ),
}
