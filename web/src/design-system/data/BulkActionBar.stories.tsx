import { Button } from "@heroui/react"
import type { Meta, StoryObj } from "@storybook/react-vite"
import { BulkActionBar } from "./BulkActionBar"

const meta = {
  title: "Design system/Data/BulkActionBar",
  component: BulkActionBar,
  args: {
    selectedCount: 3,
    allMatching: false,
    matchingTotal: 412,
    canSelectAllMatching: true,
    onSelectAllMatching: () => {},
    onClear: () => {},
    // The delete opens a `ConfirmDialog`, as every delete does; the bar holds
    // the trigger, not the confirmation.
    children: (
      <>
        <Button size="sm" variant="ghost">
          Export
        </Button>
        <Button size="sm" variant="ghost">
          Delete
        </Button>
      </>
    ),
  },
  parameters: { layout: "padded" },
} satisfies Meta<typeof BulkActionBar>

export default meta

type Story = StoryObj<typeof meta>

/**
 * A `role="toolbar"` bar that is **fixed-position** in the app, pinned to the
 * bottom of the viewport while a selection stands. Here it renders in flow, so
 * the story is readable.
 */
export const Default: Story = {}

/** One row, so the count reads in the singular. */
export const SingleSelection: Story = {
  args: {
    selectedCount: 1,
    children: (
      <Button size="sm" variant="ghost">
        Delete
      </Button>
    ),
  },
}

/**
 * The escalation: the operator has selected this page, and the bar offers the
 * whole filtered set instead. That is the point of `matchingTotal`, 412 rows
 * match the filter, 25 are on screen.
 */
export const CanSelectAllMatching: Story = {
  args: { selectedCount: 25 },
}

/** After taking that offer, the bar reports the filter rather than a page. */
export const AllMatchingSelected: Story = {
  args: { selectedCount: 412, allMatching: true },
}

/**
 * An endpoint with no total to offer, so the escalation is withheld rather than
 * promising a set whose size nobody knows.
 */
export const NoMatchingTotal: Story = {
  args: { matchingTotal: null, canSelectAllMatching: false },
}
