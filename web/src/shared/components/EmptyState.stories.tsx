import type { Meta, StoryObj } from "@storybook/react-vite"

import { UnavailableHere } from "./access/UnavailableHere"
import { EmptyState } from "./feedback/EmptyState"

const meta = {
  title: "Shared/EmptyState",
  component: EmptyState,
  args: {
    title: "No keys yet",
    description:
      "A key authenticates a caller against this gateway. Create one to make the first request.",
  },
} satisfies Meta<typeof EmptyState>

export default meta

type Story = StoryObj<typeof meta>

/** Purely informational: no action, for a list that is simply empty. */
export const Default: Story = {}

export const WithAction: Story = {
  args: { actionLabel: "Create key", onAction: () => {} },
}

/**
 * The action can be disabled while a precondition is unmet, so the panel still
 * says what to do rather than hiding the button.
 */
export const ActionDisabled: Story = {
  args: {
    description: "Add a provider first; a key with nothing to route is inert.",
    actionLabel: "Create key",
    onAction: () => {},
    isActionDisabled: true,
  },
}

/**
 * `children` slots richer content between the copy and the action. `description`
 * is deliberately typed as a string, so anything blockish goes here instead of
 * landing inside that paragraph.
 */
export const WithChildren: Story = {
  args: {
    title: "No usage yet",
    description: "Usage appears once this gateway has served a request.",
    children: (
      <ol className="flex list-decimal flex-col gap-1 pl-5 text-caption">
        <li>Add a provider credential.</li>
        <li>Create an API key.</li>
        <li>Point a client at this gateway's base URL.</li>
      </ol>
    ),
    actionLabel: "Add a provider",
    onAction: () => {},
  },
}

/**
 * `UnavailableHere` is `EmptyState` with its copy fixed: a destination this build
 * declares but does not serve. In a standalone gateway it never renders, because
 * the shell intercepts a gated-off route before the page does.
 */
export const Unavailable: Story = {
  render: () => <UnavailableHere title="Organization" />,
}
