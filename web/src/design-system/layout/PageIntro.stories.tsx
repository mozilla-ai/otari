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

/**
 * `docsHref` trails the description rather than taking the action slot, which
 * is the distinction worth keeping: the action slot is the one thing the band
 * exists to do, and a link to the manual competing for it is how a page ends up
 * with two things claiming to be that.
 */
export const WithDocsLink: Story = {
  args: {
    children: "A key authenticates a caller against this gateway.",
    docsHref: "https://example.com/docs/api-keys",
  },
}

/** Both, so the link and the primary action are visibly not the same slot. */
export const DocsLinkAndAction: Story = {
  args: {
    children: "A key authenticates a caller against this gateway.",
    docsHref: "https://example.com/docs/api-keys",
    action: <Button variant="primary">Create key</Button>,
  },
}

/**
 * `descriptionClassName` overrides the description's measure, and one caller
 * uses it: the guide, whose own prose is 560px. On the one page whose subject is
 * the measure, the widest line should not be the scanning-size paragraph
 * introducing it.
 *
 * The default measure is wider, so the two below differ only in where the
 * sentence wraps.
 */
export const NarrowerDescription: Story = {
  render: () => (
    <div className="flex flex-col gap-8">
      <div className="flex flex-col gap-1">
        <span className="text-overline">default measure</span>
        <PageIntro title="User guide">
          A reference for operating this dashboard, bundled with and
          version-matched to the running gateway.
        </PageIntro>
      </div>
      <div className="flex flex-col gap-1">
        <span className="text-overline">
          descriptionClassName="max-w-[35rem]"
        </span>
        <PageIntro title="User guide" descriptionClassName="max-w-[35rem]">
          A reference for operating this dashboard, bundled with and
          version-matched to the running gateway.
        </PageIntro>
      </div>
    </div>
  ),
}
