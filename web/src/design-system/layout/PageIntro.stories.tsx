import type { Meta, StoryObj } from "@storybook/react-vite"

import { Button } from "../actions/Button"
import { PageIntro } from "./PageIntro"

/**
 * A page's name, its one-line explanation, and the one action the page is for.
 *
 * `title` is `text-display`, the top of the ladder: nothing inside a page may
 * be larger than the page's own name. That is why a 30px KPI figure does not
 * fight a 28px title, and why a second `text-display` on one route is a bug.
 *
 * Replaces `deprecated/PageHeader`, which is still on four pages.
 */
const meta = {
  title: "Design system/Layout/PageIntro",
  component: PageIntro,
  args: { title: "API keys" },
  parameters: { layout: "padded" },
} satisfies Meta<typeof PageIntro>

export default meta

type Story = StoryObj<typeof meta>

/** Title alone, for a page whose name is the whole explanation. */
export const Default: Story = {}

export const WithDescription: Story = {
  args: {
    children:
      "A key authenticates a caller against this gateway. Each one carries its own budget and model scope.",
  },
}

/**
 * The action is the one thing the band exists to do, so it is the band's single
 * `primary`. A page with both this and a group's Save has two primaries, and
 * that is correct: each is the one thing its own band is for.
 */
export const WithAction: Story = {
  args: {
    children: "A key authenticates a caller against this gateway.",
    action: <Button variant="primary">Create key</Button>,
  },
}

/** Two actions, where the second drops to a ghost rather than competing. */
export const TwoActions: Story = {
  args: {
    children: "A key authenticates a caller against this gateway.",
    action: (
      <>
        <Button>Export CSV</Button>
        <Button variant="primary">Create key</Button>
      </>
    ),
  },
}
