import type { Meta, StoryObj } from "@storybook/react-vite"

import { ConfirmButton } from "./ConfirmButton"

const meta = {
  title: "Design system/Actions/ConfirmButton",
  component: ConfirmButton,
  args: {
    children: "Reset price",
    confirmLabel: "Reset to default",
    onConfirm: () => {},
  },
} satisfies Meta<typeof ConfirmButton>

export default meta

type Story = StoryObj<typeof meta>

/**
 * Press once to arm, again to confirm. For a destructive action that deletes
 * nothing: a regenerate, an archive, a reset to a default. A delete goes through
 * `ConfirmDialog` instead. Press it to see the armed pair.
 */
export const Default: Story = {}

/**
 * While the mutation is in flight both armed buttons are disabled, so a second
 * press cannot fire the same reset twice.
 */
export const Pending: Story = {
  args: { isPending: true },
}

/** Beside the thing it acts on, which is where it lives. */
export const InRowActions: Story = {
  render: (args) => (
    <div className="flex w-[24rem] items-center justify-between rounded-lg border border-border bg-surface px-4 py-3">
      <span className="font-mono text-caption">openai:gpt-5-mini</span>
      <ConfirmButton {...args} />
    </div>
  ),
}
