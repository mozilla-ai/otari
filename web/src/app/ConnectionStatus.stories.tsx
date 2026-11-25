import type { Meta, StoryObj } from "@storybook/react-vite"
import { ConnectionBand } from "@/app/ConnectionStatus"

/**
 * The band at the top of the content pane while the gateway is not answering,
 * and the moment after it answers again. It pushes the page down rather than
 * covering it, and pins itself while the page scrolls.
 */
const meta = {
  title: "Dashboard/Shell/ConnectionBand",
  component: ConnectionBand,
  parameters: { layout: "fullscreen" },
} satisfies Meta<typeof ConnectionBand>

export default meta

type Story = StoryObj<typeof meta>

export const Down: Story = {
  args: {
    band: {
      state: "down",
      since: new Date(2026, 10, 25, 14, 2).getTime(),
    },
  },
}

/** Shown for two seconds while the pages that failed refetch, then gone. */
export const Reconnected: Story = {
  args: { band: { state: "back" } },
}
