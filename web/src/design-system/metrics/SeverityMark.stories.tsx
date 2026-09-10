import type { Meta, StoryObj } from "@storybook/react-vite"

import { SeverityMark } from "./SeverityMark"

/**
 * A health state, as a mark and the word that carries it.
 *
 * **The word is not decoration.** A status is never color alone here, because a
 * red-green deficiency removes the only channel a bare mark has, so the word
 * ships with the mark rather than beside it as a caller's afterthought. That is
 * why `word` is a required part of the `Severity` type and not an option.
 *
 * The vocabulary is the caller's, because "alert" means different words on
 * different pages ("over", "failed", "revoked"). What the type fixes is that
 * there is one.
 */
const meta = {
  title: "Design system/Metrics/SeverityMark",
  component: SeverityMark,
  args: { severity: { status: "ok", word: "active" } },
  parameters: { layout: "padded" },
} satisfies Meta<typeof SeverityMark>

export default meta

type Story = StoryObj<typeof meta>

export const Default: Story = {}

export const EveryStatus: Story = {
  render: () => (
    <div className="flex flex-col items-start gap-2">
      <SeverityMark severity={{ status: "ok", word: "active" }} />
      <SeverityMark severity={{ status: "warn", word: "near limit" }} />
      <SeverityMark severity={{ status: "alert", word: "over budget" }} />
    </div>
  ),
}

export const EveryStatusDark: Story = {
  render: () => (
    <div className="flex flex-col items-start gap-2">
      <SeverityMark severity={{ status: "ok", word: "active" }} />
      <SeverityMark severity={{ status: "warn", word: "near limit" }} />
      <SeverityMark severity={{ status: "alert", word: "over budget" }} />
    </div>
  ),
  globals: { theme: "dark" },
}

/** The same three statuses wearing each page's own vocabulary. */
export const Vocabularies: Story = {
  render: () => (
    <div className="flex flex-col items-start gap-2">
      <SeverityMark severity={{ status: "alert", word: "revoked" }} />
      <SeverityMark severity={{ status: "alert", word: "failed" }} />
      <SeverityMark severity={{ status: "warn", word: "retrying" }} />
      <SeverityMark severity={{ status: "warn", word: "degraded" }} />
      <SeverityMark severity={{ status: "ok", word: "verified" }} />
    </div>
  ),
}
