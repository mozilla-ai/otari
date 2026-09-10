import type { Meta, StoryObj } from "@storybook/react-vite"

import { EmptyMessage } from "./EmptyMessage"

/**
 * One line where rows would have been.
 *
 * The smaller sibling of `EmptyState`, and the line between them is whether
 * there is anything to *do*: `EmptyState` is a panel with a title and usually an
 * action, for a page whose main content is empty. This is a sentence inside a
 * band that has other content around it, and it offers nothing, because the
 * answer is somewhere else on the page.
 */
const meta = {
  title: "Design system/Feedback/EmptyMessage",
  component: EmptyMessage,
  args: { children: "No requests in this window." },
  parameters: { layout: "padded" },
} satisfies Meta<typeof EmptyMessage>

export default meta

type Story = StoryObj<typeof meta>

export const Default: Story = {
  render: () => (
    <div className="w-96 border-y border-border">
      <EmptyMessage>No requests in this window.</EmptyMessage>
    </div>
  ),
}

/**
 * `minHeightClass` holds a band open that must not collapse, in `rem`. The
 * height *is* the space, so the padding comes off with it: a chart's band keeps
 * the chart's height whether or not there is data, and the page does not jump
 * when the first row arrives.
 */
export const HoldingABandOpen: Story = {
  render: () => (
    <div className="flex w-96 flex-col gap-6">
      <div className="flex flex-col gap-1">
        <span className="text-overline">no minimum</span>
        <div className="border-y border-border">
          <EmptyMessage>No requests in this window.</EmptyMessage>
        </div>
      </div>
      <div className="flex flex-col gap-1">
        <span className="text-overline">
          min-h-[12rem], a chart's own height
        </span>
        <div className="border-y border-border">
          <EmptyMessage minHeightClass="min-h-[12rem]">
            No requests in this window.
          </EmptyMessage>
        </div>
      </div>
    </div>
  ),
}
