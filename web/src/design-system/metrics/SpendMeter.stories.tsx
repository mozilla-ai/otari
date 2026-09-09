import type { Meta, StoryObj } from "@storybook/react-vite"

import { SpendMeter, spendState } from "./SpendMeter"

/**
 * Money against an allocation, with the state it implies.
 *
 * The difference from `Meter` is `spendState`, exported beside it: the same
 * three thresholds decide the bar's color here and the word a row prints
 * elsewhere, so a budget that reads "near limit" in a table cannot be drawn
 * on-track in its meter. That is the whole reason the classifier ships with the
 * component instead of living at a call site.
 */
const meta = {
  title: "Design system/Metrics/SpendMeter",
  component: SpendMeter,
  args: {
    spent: 412.08,
    allocated: 1000,
    ariaLabel: "$412.08 of $1,000 spent",
  },
  parameters: { layout: "padded" },
} satisfies Meta<typeof SpendMeter>

export default meta

type Story = StoryObj<typeof meta>

export const Default: Story = {
  render: () => (
    <div className="w-64">
      <SpendMeter spent={412.08} allocated={1000} ariaLabel="$412 of $1,000" />
    </div>
  ),
}

/**
 * The three states, with `spendState`'s own answer printed beside each so the
 * classifier and the bar can be checked against one another.
 */
export const States: Story = {
  render: () => (
    <div className="flex w-72 flex-col gap-4">
      {[
        [412.08, 1000],
        [860, 1000],
        [1041.2, 1000],
      ].map(([spent, allocated]) => (
        <div key={spent} className="flex flex-col gap-1">
          <span className="text-caption">
            {`$${spent} of $${allocated} — spendState: ${spendState(spent, allocated)}`}
          </span>
          <SpendMeter
            spent={spent}
            allocated={allocated}
            ariaLabel={`$${spent} of $${allocated} spent`}
          />
        </div>
      ))}
    </div>
  ),
}

/**
 * `nearLimitAt` moves the warning threshold, which a deployment with a tighter
 * tolerance wants. The default is 0.8.
 */
export const CustomThreshold: Story = {
  render: () => (
    <div className="flex w-72 flex-col gap-4">
      {[0.8, 0.5].map((threshold) => (
        <div key={threshold} className="flex flex-col gap-1">
          <span className="text-caption">
            {`nearLimitAt ${threshold} — spendState: ${spendState(600, 1000, threshold)}`}
          </span>
          <SpendMeter
            spent={600}
            allocated={1000}
            nearLimitAt={threshold}
            ariaLabel="$600 of $1,000 spent"
          />
        </div>
      ))}
    </div>
  ),
}

/**
 * No allocation, which is an unbounded budget rather than a full one. It
 * classifies as on-track, because a division by zero would otherwise report
 * every unlimited budget as over.
 */
export const Unallocated: Story = {
  render: () => (
    <div className="w-64">
      <SpendMeter
        spent={412.08}
        allocated={0}
        ariaLabel="$412.08 spent, no limit"
      />
    </div>
  ),
}
