import type { Meta, StoryObj } from "@storybook/react-vite"

import { Meter } from "./Meter"

/**
 * A proportion, as a bar.
 *
 * `ariaLabel` is required, and motion-and-access.md gives the reason in four
 * words: a bar has no text. Without a name it is a decoration that a screen
 * reader cannot read a value from.
 *
 * Reach for `SpendMeter` where the proportion is money against an allocation,
 * because that one classifies the state as well as drawing it. Reach for
 * `ProgressBar` where the thing is *finishing* rather than *filling*.
 */
const meta = {
  title: "Design system/Metrics/Meter",
  component: Meter,
  args: { fraction: 0.41, ariaLabel: "41% of the budget used" },
  parameters: { layout: "padded" },
} satisfies Meta<typeof Meter>

export default meta

type Story = StoryObj<typeof meta>

export const Default: Story = {
  render: () => (
    <div className="w-64">
      <Meter fraction={0.41} ariaLabel="41% used" />
    </div>
  ),
}

/** Across the range, including the two ends a caller can hand it. */
export const Range: Story = {
  render: () => (
    <div className="flex w-64 flex-col gap-3">
      {[0, 0.05, 0.41, 0.8, 1].map((fraction) => (
        <div key={fraction} className="flex flex-col gap-1">
          <span className="text-caption">{`${Math.round(fraction * 100)}%`}</span>
          <Meter
            fraction={fraction}
            ariaLabel={`${Math.round(fraction * 100)}% used`}
          />
        </div>
      ))}
    </div>
  ),
}

/** Over 1, which a caller can produce and the bar must not draw past its track. */
export const Overfilled: Story = {
  render: () => (
    <div className="w-64">
      <Meter fraction={1.4} ariaLabel="140% of the allocation used" />
    </div>
  ),
}
