import type { Meta, StoryObj } from "@storybook/react-vite"

import { ConfirmRowAction } from "./ConfirmRowAction"
import { RowAction, RowActionRow } from "./RowAction"

/**
 * The two-step destructive confirm, inside a table row.
 *
 * `ConfirmButton`'s sibling for a row. It supplies its own danger styling and
 * its own Cancel, so a call site cannot paint the hue at rest.
 *
 * **The Cancel that appears when armed is load-bearing. Do not simplify it
 * away.** The escalation is hue-only, and hue is the one channel a red-green
 * deficiency removes: measured, the ghost edge against the danger edge is
 * 1.17:1 in light, 1.11:1 in dark, and 1.03:1 under simulated protanopia. What
 * survives is structural, a second control appearing and the row's layout
 * changing, which no color deficiency hides. Press one below to see it.
 */
const meta = {
  title: "Design system/Actions/ConfirmRowAction",
  component: ConfirmRowAction,
  args: {
    confirmLabel: "Revoke permanently",
    onConfirm: () => {},
    children: "Revoke",
  },
  parameters: { layout: "padded" },
} satisfies Meta<typeof ConfirmRowAction>

export default meta

type Story = StoryObj<typeof meta>

/** Press it once to arm, and watch a second control appear beside it. */
export const Default: Story = {}

/**
 * The two labels say different things: the trigger names the object, the armed
 * confirm names the consequence. The same word twice tells an operator nothing
 * about what changed.
 */
export const InALane: Story = {
  render: () => (
    <RowActionRow>
      <RowAction onPress={() => {}}>Edit</RowAction>
      <ConfirmRowAction confirmLabel="Revoke permanently" onConfirm={() => {}}>
        Revoke
      </ConfirmRowAction>
    </RowActionRow>
  ),
}

/** `isPending` while the mutation is in flight, after the second press. */
export const Pending: Story = { args: { isPending: true } }
