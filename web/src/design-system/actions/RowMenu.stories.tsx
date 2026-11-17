import type { Meta, StoryObj } from "@storybook/react-vite"
import { FiDollarSign, FiEdit2, FiTrash2, FiUserMinus } from "react-icons/fi"

import { RowMenu } from "./RowMenu"
import { RowMenuItemContent } from "./RowMenuItemContent"

/**
 * A row's actions behind one three-dot trigger, built from data. Each action is
 * spelled out with what it does, and a `hint` under its label says the
 * consequence or the current state. `RowActionRow` draws the same trigger and
 * menu from `RowAction` children; this is for a row whose menu wants a heading
 * above its actions or a sentence under each.
 *
 * A refused action stays, disabled, with its reason as its hint, and the
 * trigger's name carries the reason too: a disabled item takes no focus.
 */
const meta = {
  title: "Design system/Actions/RowMenu",
  component: RowMenu,
  args: {
    label: "Actions for Wei Zhang",
    actions: [
      {
        id: "edit",
        label: "Change role…",
        hint: "Admin today",
        icon: FiEdit2,
        run: () => {},
      },
      {
        id: "limit",
        label: "Set a spend limit…",
        hint: "Uses the workspace default",
        icon: FiDollarSign,
        run: () => {},
      },
      {
        id: "remove",
        label: "Remove…",
        hint: "Their keys stop working",
        icon: FiUserMinus,
        isDestructive: true,
        run: () => {},
      },
    ],
  },
  parameters: { layout: "centered" },
} satisfies Meta<typeof RowMenu>

export default meta

type Story = StoryObj<typeof meta>

export const Default: Story = {}

/** The last owner cannot be removed, and the trigger says so. */
export const WithARefusal: Story = {
  args: {
    actions: [
      {
        id: "remove",
        label: "Remove…",
        hint: "This is the last active owner. Promote someone else first.",
        icon: FiUserMinus,
        isDisabled: true,
        run: () => {},
      },
    ],
  },
}

/** What the row is, above its actions, and a line under them. */
export const WithHeadAndFoot: Story = {
  args: {
    head: (
      <div className="border-b border-border px-3 py-3 text-body">
        checkout-service
      </div>
    ),
    foot: (
      <p className="border-t border-border px-3 py-2 text-caption">
        Owned by another workspace.
      </p>
    ),
  },
}

/** One item's body, plain and destructive, as both menus draw it. */
export const ItemBodies: Story = {
  render: () => (
    <ul className="flex w-[19rem] flex-col">
      <li className="flex items-start gap-2.5 px-3 py-2.5">
        <RowMenuItemContent
          icon={FiEdit2}
          label="Edit…"
          hint="Its name and the models it may call"
          isRed={false}
        />
      </li>
      <li className="flex items-start gap-2.5 border-t border-border-subtle px-3 py-2.5">
        <RowMenuItemContent
          icon={FiTrash2}
          label="Delete…"
          hint={undefined}
          isRed
        />
      </li>
    </ul>
  ),
}
