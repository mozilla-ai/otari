import type { Meta, StoryObj } from "@storybook/react-vite"

import { Button } from "../actions/Button"
import { PageIntro } from "./PageIntro"

/**
 * A page's name and the one action the page is for. No sentence explains the
 * page under its title; `detail` is a line about the record the page shows.
 *
 * `title` is `text-display`, the top of the ladder: nothing inside a page may
 * be larger than the page's own name. That is why a 30px KPI figure does not
 * fight a 28px title, and why a second `text-display` on one route is a bug.
 *
 * Replaced `deprecated/PageHeader`, which is gone: every page opens with this.
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

/** A record's own facts under its name: the one line a page keeps. */
export const WithDetail: Story = {
  args: { title: "Wei Zhang", detail: "Admin · wei@example.com" },
}

/**
 * The action is the one thing the band exists to do, so it is the band's single
 * `primary`. A page with both this and a group's Save has two primaries, and
 * that is correct: each is the one thing its own band is for.
 */
export const WithLeadingMark: Story = {
  args: {
    title: "Sentry",
    leading: (
      <span
        aria-hidden
        className="flex size-11 items-center justify-center rounded-md border border-control-border bg-surface-alt text-title text-muted"
      >
        S
      </span>
    ),
  },
}

export const WithAction: Story = {
  args: {
    action: <Button variant="primary">Create key</Button>,
  },
}

/** Two actions, where the second drops to a ghost rather than competing. */
export const TwoActions: Story = {
  args: {
    action: (
      <>
        <Button>Export CSV</Button>
        <Button variant="primary">Create key</Button>
      </>
    ),
  },
}
