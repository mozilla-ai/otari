import type { Meta, StoryObj } from "@storybook/react-vite"

import { Skeleton } from "./Skeleton"

/**
 * A placeholder holding the space a value will take.
 *
 * The rule for reaching for one is not "show something while loading": a
 * skeleton is right when the shape of what is coming is *known*, and wrong when
 * it is not, because one that guesses wrong moves the page twice instead of
 * once. Where the shape is unknown, `PageLoading` is the honest answer.
 *
 * The pulse is guarded with `motion-reduce:animate-none`, like every other
 * animation here: this one runs until data arrives, which is exactly the kind a
 * vestibular disorder cannot dismiss.
 */
const meta = {
  title: "Design system/Feedback/Skeleton",
  component: Skeleton,
  parameters: { layout: "padded" },
} satisfies Meta<typeof Skeleton>

export default meta

type Story = StoryObj<typeof meta>

/** The default is one line of body text, which is the most common case. */
export const Default: Story = {
  render: () => (
    <div className="w-64">
      <Skeleton />
    </div>
  ),
}

/** The shape it is actually for: a strip whose four cells are known in advance. */
export const KpiStrip: Story = {
  render: () => (
    <div className="grid w-[40rem] grid-cols-4 gap-4">
      {["spend", "requests", "tokens", "errors"].map((cell) => (
        <div key={cell} className="flex flex-col gap-2">
          <Skeleton className="h-3 w-16" />
          <Skeleton className="h-8 w-24" />
        </div>
      ))}
    </div>
  ),
}

/** A table whose row count is known because the page size is. */
export const TableRows: Story = {
  render: () => (
    <div className="flex w-[40rem] flex-col gap-2">
      {[0, 1, 2, 3, 4].map((row) => (
        <div
          key={row}
          className="flex items-center gap-4 border-b border-border pb-2"
        >
          <Skeleton className="h-4 w-48" />
          <Skeleton className="h-4 w-20" />
          <Skeleton className="ml-auto h-4 w-16" />
        </div>
      ))}
    </div>
  ),
}

/**
 * `ariaLabel` is for the rare skeleton that is the only thing on screen.
 * Without it the box is hidden from assistive tech, which is right when a
 * region around it already says it is loading: announcing eight "loading" boxes
 * for one table is worse than silence.
 */
export const Announced: Story = {
  render: () => (
    <div className="w-64">
      <Skeleton ariaLabel="Loading usage totals" className="h-8 w-full" />
    </div>
  ),
}
