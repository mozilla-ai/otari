import type { Meta, StoryObj } from "@storybook/react-vite"

import { PasswordCard } from "./PasswordCard"

/**
 * Claim a password, or change one.
 *
 * Which of the two the card offers is read from the bootstrap's
 * `sign_in_methods`, not from a prop: a deployment still listing `master_key` has
 * not been claimed, so the card offers to claim it, and once it lists `password`
 * instead the same card becomes a change-password form. That makes
 * `parameters.deployment` the switch for these stories (see
 * `.storybook/appContext.tsx`).
 *
 * Claiming is one-way, and it retires the master key as a sign-in for the whole
 * deployment, which is why it is a card of its own rather than a field on a
 * settings page.
 */
const meta = {
  title: "Dashboard/Account/PasswordCard",
  component: PasswordCard,
  parameters: { layout: "padded" },
} satisfies Meta<typeof PasswordCard>

export default meta

type Story = StoryObj<typeof meta>

/**
 * Unclaimed: the deployment still signs in with the master key, so the card offers
 * to claim it with an email and password.
 */
export const Unclaimed: Story = {
  parameters: { deployment: { sign_in_methods: ["master_key"] } },
  render: () => (
    <div className="w-[40rem]">
      <PasswordCard />
    </div>
  ),
}

/** Claimed: the same card is now a change-password form. */
export const Claimed: Story = {
  parameters: { deployment: { sign_in_methods: ["password"] } },
  render: () => (
    <div className="w-[40rem]">
      <PasswordCard />
    </div>
  ),
}

/**
 * Mid-migration, where the gateway accepts both. The card treats any list without
 * `master_key` as claimed, so this still reads as unclaimed, the master key
 * works, therefore it can still be retired.
 */
export const BothMethods: Story = {
  parameters: { deployment: { sign_in_methods: ["master_key", "password"] } },
  render: () => (
    <div className="w-[40rem]">
      <PasswordCard />
    </div>
  ),
}
