import type { Meta, StoryObj } from "@storybook/react-vite"

import { ConfirmButton } from "./ConfirmButton"

const meta = {
  title: "Design system/Actions/ConfirmButton",
  component: ConfirmButton,
  args: {
    children: "Revoke",
    confirmLabel: "Revoke key",
    onConfirm: () => {},
  },
} satisfies Meta<typeof ConfirmButton>

export default meta

type Story = StoryObj<typeof meta>

/**
 * Press once to arm, again to confirm. Two clicks instead of a modal, which is
 * what keeps a revoke or delete in a table row from pulling a dialog in with it.
 * Press it to see the armed pair.
 */
export const Default: Story = {}

/**
 * While the mutation is in flight both armed buttons are disabled, so a second
 * press cannot fire the same revoke twice.
 */
export const Pending: Story = {
  args: { isPending: true },
}

/** In a row of actions, which is where it lives. */
export const InRowActions: Story = {
  render: (args) => (
    <div className="flex w-[24rem] items-center justify-between rounded-lg border border-border bg-surface px-4 py-3">
      <span className="font-mono text-caption">otari_sk_…4f2a</span>
      <ConfirmButton {...args} />
    </div>
  ),
}
