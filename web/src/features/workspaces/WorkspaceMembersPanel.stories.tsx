import type { Meta, StoryObj } from "@storybook/react-vite"

import { organizationMember, workspaceMember } from "@/tests/fixtures"

import { WorkspaceMembersPanel } from "./WorkspaceMembersPanel"

/**
 * One workspace's roster, plus the form that adds to it.
 *
 * It takes an id and a name rather than a `Workspace`, because the Members page
 * reaches it holding only the caller's own membership, which carries both and
 * nothing else. The organization roster arrives as a prop (the page already has
 * it); only the workspace roster is this component's own query.
 *
 * `canManageWorkspace` is the whole difference between a roster you can edit and
 * one you can only read, so both are worth seeing.
 */
const WORKSPACE_ID = "44444444-4444-4444-4444-444444444444"

const ORG_MEMBERS = [
  organizationMember({
    organization_member_id: "22222222-2222-2222-2222-222222222201",
    user_id: "33333333-3333-3333-3333-333333333301",
    attribution_user_id: "33333333-3333-3333-3333-333333333301",
    email: "ada@example.com",
    full_name: "Ada Lovelace",
    role: "owner",
  }),
  organizationMember({
    organization_member_id: "22222222-2222-2222-2222-222222222202",
    user_id: "33333333-3333-3333-3333-333333333302",
    attribution_user_id: "33333333-3333-3333-3333-333333333302",
    email: "grace@example.com",
    full_name: "Grace Hopper",
    role: "member",
  }),
  organizationMember({
    organization_member_id: "22222222-2222-2222-2222-222222222203",
    user_id: "33333333-3333-3333-3333-333333333303",
    attribution_user_id: "33333333-3333-3333-3333-333333333303",
    email: "alan@example.com",
    full_name: "Alan Turing",
    role: "member",
  }),
]

const ROSTER = [
  workspaceMember({
    id: "55555555-5555-5555-5555-555555555501",
    user_id: "33333333-3333-3333-3333-333333333301",
    role: "owner",
  }),
  workspaceMember({
    id: "55555555-5555-5555-5555-555555555502",
    user_id: "33333333-3333-3333-3333-333333333302",
    role: "member",
  }),
]

const membersPath = `/v1/workspaces/${WORKSPACE_ID}/members`

const meta = {
  title: "Dashboard/Workspaces/WorkspaceMembersPanel",
  component: WorkspaceMembersPanel,
  args: {
    workspaceId: WORKSPACE_ID,
    workspaceName: "Platform",
    orgMembers: ORG_MEMBERS,
    rosterResolved: true,
    canManageWorkspace: true,
  },
  parameters: { api: { [membersPath]: ROSTER }, layout: "padded" },
} satisfies Meta<typeof WorkspaceMembersPanel>

export default meta

type Story = StoryObj<typeof meta>

/**
 * A manageable roster: roles are editable, members removable, and the add form
 * offers the organization members who are not in it yet (Alan, here).
 */
export const CanManage: Story = {
  render: (args) => (
    <div className="w-[44rem]">
      <WorkspaceMembersPanel {...args} />
    </div>
  ),
}

/** A member's view: the same roster, no controls. */
export const ReadOnly: Story = {
  args: { canManageWorkspace: false },
  render: (args) => (
    <div className="w-[44rem]">
      <WorkspaceMembersPanel {...args} />
    </div>
  ),
}

/**
 * Every organization member is already in the workspace, so the add form has
 * nothing left to offer.
 */
export const EveryoneAlreadyIn: Story = {
  args: { orgMembers: ORG_MEMBERS.slice(0, 2) },
  render: (args) => (
    <div className="w-[44rem]">
      <WorkspaceMembersPanel {...args} />
    </div>
  ),
}

/** A brand-new workspace, whose only member is whoever created it. */
export const SingleOwner: Story = {
  parameters: { api: { [membersPath]: [ROSTER[0]] } },
  render: (args) => (
    <div className="w-[44rem]">
      <WorkspaceMembersPanel {...args} />
    </div>
  ),
}

/**
 * The organization roster has not resolved yet. `rosterResolved` is a prop rather
 * than something this component waits on, because the page owns that query, and
 * an unresolved roster must not read as "nobody to add".
 */
export const RosterNotResolved: Story = {
  args: { rosterResolved: false, orgMembers: [] },
  render: (args) => (
    <div className="w-[44rem]">
      <WorkspaceMembersPanel {...args} />
    </div>
  ),
}
