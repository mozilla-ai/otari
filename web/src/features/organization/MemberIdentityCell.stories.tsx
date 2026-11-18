import type { Meta, StoryObj } from "@storybook/react-vite"

import { organizationMember } from "@/tests/fixtures"
import { MemberIdentityCell } from "./MemberIdentityCell"

/**
 * The Members table's first column.
 *
 * The four stories are the four shapes of the second line: an address, the mark
 * that stands in for one, a qualifier alone, or nothing.
 */
const meta = {
  title: "Dashboard/Organization/MemberIdentityCell",
  component: MemberIdentityCell,
  args: { access: { text: "All models", tone: "muted" } },
  parameters: { layout: "padded" },
} satisfies Meta<typeof MemberIdentityCell>

export default meta

type Story = StoryObj<typeof meta>

export const WithAnAddress: Story = {
  args: {
    member: organizationMember({
      user_id: "33333333-3333-3333-3333-333333333302",
      full_name: "Grace Hopper",
      email: "grace.hopper@example.com",
    }),
  },
}

export const AdmittedThroughSso: Story = {
  args: {
    member: organizationMember({
      user_id: "33333333-3333-3333-3333-333333333301",
      full_name: "Ada Lovelace",
      email: null,
      is_sso_account: true,
    }),
  },
}

/** An unclaimed operator identity: no address, and no connection behind it. */
export const Unclaimed: Story = {
  args: {
    member: organizationMember({
      user_id: "33333333-3333-3333-3333-333333333303",
      full_name: "Operator",
      email: null,
    }),
  },
}

/** What a reader who does not operate the deployment sees. */
export const WithoutTheAccessQualifier: Story = {
  args: {
    access: undefined,
    member: organizationMember({
      user_id: "33333333-3333-3333-3333-333333333305",
      full_name: "Katherine Johnson",
      email: null,
      is_sso_account: true,
    }),
  },
}
