import type { Meta, StoryObj } from "@storybook/react-vite"

import { API_ROOT } from "@/shared/api/client"
import { organizationContext } from "@/tests/fixtures"

import { PasswordCard } from "./PasswordCard"

/**
 * Set a password, claim a deployment with one, or change one.
 *
 * Which of the three the card offers is read from the *caller* on the
 * membership context and not from a prop: `has_password` says whether a current
 * password has to be proved, and the address says whether one has to be
 * supplied. Those are the two conditions `PUT /v1/auth/password` branches on, so
 * `parameters.api` is the switch for these stories.
 *
 * The form itself is a dialog, which is what the card's action opens. Claiming
 * is one-way and retires the master key as a sign-in for the whole deployment,
 * which is why it is a card of its own rather than a field on a settings page.
 */
const meta = {
  title: "Dashboard/Account/PasswordCard",
  component: PasswordCard,
  parameters: { layout: "padded" },
} satisfies Meta<typeof PasswordCard>

export default meta

type Story = StoryObj<typeof meta>

function contextFor(email: string | null, hasPassword: boolean) {
  return {
    api: {
      [`${API_ROOT}/organizations/me`]: organizationContext({
        caller: {
          user_id: "33333333-3333-3333-3333-333333333333",
          email,
          full_name: "Ada Lovelace",
          has_password: hasPassword,
        },
      }),
    },
  }
}

/**
 * Unclaimed: first boot leaves the operator with no address and no password, so
 * the card offers to claim the deployment with both.
 */
export const Unclaimed: Story = {
  parameters: {
    ...contextFor(null, false),
    deployment: { sign_in_methods: ["master_key"] },
  },
}

/**
 * Signed in through GitHub, Google or a passkey, so there is an address and no
 * password. The form asks for the new password alone, which is what somebody in
 * this state could not reach before (mozilla-ai/otari-ai#2099).
 */
export const NoPasswordYet: Story = {
  parameters: contextFor("ada@example.com", false),
}

/** The steady state: replacing a password means proving the current one. */
export const HasPassword: Story = {
  parameters: contextFor("ada@example.com", true),
}

/**
 * The membership context could not be read, so the card offers no form rather
 * than one built on a guess about which credential the caller holds.
 */
export const IdentityUnavailable: Story = {
  parameters: {
    api: {
      [`${API_ROOT}/organizations/me`]: {
        $status: 503,
        $body: { detail: "The control plane is unreachable." },
      },
    },
  },
}
