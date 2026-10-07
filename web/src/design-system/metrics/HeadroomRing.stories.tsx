import type { Meta, StoryObj } from "@storybook/react-vite"

import { HeadroomRing } from "./HeadroomRing"

/**
 * What is left of a limit, as a ring that empties as it is used, with the
 * figure beside it.
 *
 * Reach for it where a table cell has to say how much room remains and a bar
 * would not fit. Reach for `SpendMeter` where there is width for a bar and the
 * money figures beside it. Both classify through `spendState`.
 */
const meta = {
  title: "Design system/Metrics/HeadroomRing",
  component: HeadroomRing,
  args: { used: 0.41 },
  parameters: { layout: "padded" },
} satisfies Meta<typeof HeadroomRing>

export default meta

type Story = StoryObj<typeof meta>

export const Default: Story = {}

/** Across the range: untouched, on track, near the limit, at it, past it. */
export const Range: Story = {
  render: () => (
    <div className="flex flex-col gap-3">
      {[0, 0.41, 0.75, 0.88, 0.999, 1, 1.37].map((used) => (
        <div key={used} className="flex items-center gap-4">
          <span className="w-24 text-caption">{`used ${used}`}</span>
          <HeadroomRing used={used} />
        </div>
      ))}
    </div>
  ),
}

/**
 * `nearLimitAt` moves the point where the arc turns danger, as it does on
 * `SpendMeter`. The default is 0.8; here 0.6 used is already near at 0.5.
 */
export const CustomThreshold: Story = {
  render: () => (
    <div className="flex flex-col gap-3">
      {[0.8, 0.5].map((threshold) => (
        <div key={threshold} className="flex items-center gap-4">
          <span className="w-32 text-caption">{`nearLimitAt ${threshold}`}</span>
          <HeadroomRing used={0.6} nearLimitAt={threshold} />
        </div>
      ))}
    </div>
  ),
}
