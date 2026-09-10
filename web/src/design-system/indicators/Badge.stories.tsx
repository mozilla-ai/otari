import type { Meta, StoryObj } from "@storybook/react-vite"

import { Badge } from "./Badge"

/**
 * A fact a row states about itself, as a dot and a word.
 *
 * Two tones and no more: `muted` states a fact (which provider, which scope,
 * whether a credential is set), `warn` says the row is not doing what its
 * neighbors are.
 *
 * Not uppercased and not mono, unlike the status marks it shares a dot with.
 * Those draw from a fixed vocabulary of a word or two; this takes whatever a
 * caller passes, and its callers pass sentences. Uppercasing a sentence is
 * shouting, and mono is for values you copy rather than prose you read.
 *
 * Reach for `Chip` instead where the label sits on a fill, and for
 * `SeverityMark` where the row is healthy or broken rather than merely
 * different.
 */
const meta = {
  title: "Design system/Indicators/Badge",
  component: Badge,
  args: { tone: "muted", children: "Every workspace, including new ones" },
  parameters: { layout: "padded" },
} satisfies Meta<typeof Badge>

export default meta

type Story = StoryObj<typeof meta>

export const Default: Story = {}

export const Tones: Story = {
  render: () => (
    <div className="flex flex-col items-start gap-2">
      <Badge tone="muted">Every workspace, including new ones</Badge>
      <Badge tone="warn">No credential stored</Badge>
    </div>
  ),
}

/** The shape it ships in: one line under a card's own title. */
export const InContext: Story = {
  render: () => (
    <div className="flex w-96 flex-col gap-1 border border-border p-3">
      <span className="text-title">Web search</span>
      <Badge tone="warn">No credential stored, so requests are refused</Badge>
    </div>
  ),
}
