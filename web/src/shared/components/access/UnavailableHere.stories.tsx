import type { Meta, StoryObj } from "@storybook/react-vite"

import { UnavailableHere } from "./UnavailableHere"

/**
 * `EmptyState` with its copy fixed, for a destination this build declares but
 * does not serve.
 *
 * It sits in the application layer rather than the design system because
 * "unavailable here" is a fact about which deployment answered the page, which
 * is a surface, and the design system does not know that deployments exist.
 * In a standalone gateway it never renders at all: the shell intercepts a
 * gated-off route before the page does.
 */
const meta = {
  title: "Dashboard/Access/UnavailableHere",
  component: UnavailableHere,
  args: { title: "Organization" },
} satisfies Meta<typeof UnavailableHere>

export default meta

type Story = StoryObj<typeof meta>

export const Default: Story = {}

/** The hosted surfaces an operator most often reaches for from a standalone build. */
export const HostedSurfaces: Story = {
  render: () => (
    <div className="flex flex-col gap-8">
      <UnavailableHere title="Organization" />
      <UnavailableHere title="Accounts" />
    </div>
  ),
}
