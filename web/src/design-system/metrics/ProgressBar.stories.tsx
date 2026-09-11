import type { Meta, StoryObj } from "@storybook/react-vite"

import { ProgressBar } from "./ProgressBar"

/**
 * A task on its way to finishing.
 *
 * Not `Meter`, and the difference is visible to an operator: a meter at 100%
 * says "you are at your limit", a progress bar at 100% says "finished". A meter
 * is a measurement against a limit that sits there indefinitely; this describes
 * work in flight and disappears once it is done.
 */
const meta = {
  title: "Design system/Metrics/ProgressBar",
  component: ProgressBar,
  args: { label: "Refreshing prices", value: 30 },
  parameters: { layout: "padded" },
} satisfies Meta<typeof ProgressBar>

export default meta

type Story = StoryObj<typeof meta>

export const Default: Story = {
  render: () => (
    <div className="w-72">
      <ProgressBar label="Refreshing prices" value={30} />
    </div>
  ),
}

/**
 * `valueLabel` is a string rather than a formatter's output, because only the
 * caller knows the unit, and a count says more than a percentage about work
 * somebody is waiting on. It is mono and tabular so the line does not jitter as
 * the number ticks up.
 */
export const WithCount: Story = {
  render: () => (
    <div className="flex w-72 flex-col gap-4">
      <ProgressBar
        label="Refreshing prices"
        value={120}
        max={400}
        valueLabel="120 of 400 models"
      />
      <ProgressBar
        label="Re-encrypting credentials"
        value={7}
        max={9}
        valueLabel="7 of 9 providers"
      />
    </div>
  ),
}

/** Across the range. */
export const Range: Story = {
  render: () => (
    <div className="flex w-72 flex-col gap-4">
      {[0, 25, 60, 100].map((value) => (
        <ProgressBar
          key={value}
          label={`At ${value}%`}
          value={value}
          valueLabel={`${value}%`}
        />
      ))}
    </div>
  ),
}

/**
 * For work whose total is unknown. It animates rather than filling, so it must
 * not be paired with a `valueLabel` claiming a count it cannot know.
 */
export const Indeterminate: Story = {
  render: () => (
    <div className="w-72">
      <ProgressBar label="Discovering models" isIndeterminate />
    </div>
  ),
}
