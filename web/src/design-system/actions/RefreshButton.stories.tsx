import type { Meta, StoryObj } from "@storybook/react-vite"

import { RefreshButton } from "./RefreshButton"

const meta = {
  title: "Design system/Actions/RefreshButton",
  component: RefreshButton,
  args: { onRefresh: () => {} },
} satisfies Meta<typeof RefreshButton>

export default meta

type Story = StoryObj<typeof meta>

/**
 * Before the first successful load. `updatedAt` is a TanStack Query
 * `dataUpdatedAt`, which is 0 until then; that reads as "never", so the freshness
 * label is hidden rather than claiming an epoch timestamp.
 */
export const NeverLoaded: Story = {}

/**
 * With a timestamp, so an operator can tell stale numbers from fresh ones. The
 * label is kept current by a display clock that ticks every 15s and pauses while
 * the tab is hidden, it fetches nothing.
 */
export const RecentlyUpdated: Story = {
  args: { updatedAt: Date.now() - 42_000 },
}

export const Stale: Story = {
  args: { updatedAt: Date.now() - 1_000 * 60 * 37 },
}

/** The icon spins while a refetch is in flight, and the button is disabled. */
export const Fetching: Story = {
  args: { isFetching: true, updatedAt: Date.now() - 90_000 },
}
