import type { Meta, StoryObj } from "@storybook/react-vite"

import { Avatar } from "./Avatar"

/**
 * A person or a workspace, as a monogram.
 *
 * It takes the letters rather than deriving them. Turning an identity into two
 * letters is application logic: this product falls back from a full name to an
 * email address and splits the two differently, and `AccountMenu`'s
 * `initialsFor` carries all of it. A derivation here would be the same rules
 * spelled worse, in a layer with no business knowing an account has an email.
 *
 * `aria-hidden`, because in every place it renders the name is on screen beside
 * it. That is why no story below has an accessible name to show.
 */
const meta = {
  title: "Design system/Indicators/Avatar",
  component: Avatar,
  args: { initials: "AL" },
  parameters: { layout: "padded" },
} satisfies Meta<typeof Avatar>

export default meta

type Story = StoryObj<typeof meta>

export const Default: Story = {}

/** 20px and 26px. The 9px monogram is off-scale on purpose (typography.md). */
export const Sizes: Story = {
  render: () => (
    <div className="flex items-center gap-3">
      <Avatar initials="AL" size="sm" />
      <Avatar initials="AL" size="md" />
    </div>
  ),
}

/**
 * The shape it ships in: the account row in the rail, which is where this
 * markup came from.
 */
export const InContext: Story = {
  render: () => (
    <div className="flex w-64 items-center gap-2 border border-border p-2">
      <Avatar initials="AL" />
      <span className="min-w-0 flex-1 truncate text-shell-label text-foreground">
        ada.lovelace@mozilla.ai
      </span>
    </div>
  ),
}

/** The no-identity case the shell falls back to when nobody can be named. */
export const Unknown: Story = { args: { initials: "··" } }
