import type { Meta, StoryObj } from "@storybook/react-vite"
import { API_ROOT } from "@/shared/api/client"
import {
  organizationContext,
  workspace,
  workspaceCodeExecutionPolicy,
} from "@/tests/fixtures"
import { WorkspaceCodeExecutionPolicyCard } from "./WorkspaceCodeExecutionPolicyCard"

/**
 * Whether a workspace may run code: one switch over three stored states. No
 * policy reads as allowed, because it behaves exactly like one.
 *
 * This card reads the *selected* workspace from context rather than a prop, so
 * these stories mock /api/v1/organizations/me, that is what
 * `SelectedWorkspaceProvider` seeds itself from (see `.storybook/appContext.tsx`).
 */
const WORKSPACE_ID = "44444444-4444-4444-4444-444444444444"

// A membership on the context is the switcher's own shape (id, name, role), not a
// full `WorkspaceMember` row, so this is written out rather than built from the
// `workspaceMember` fixture.
const CONTEXT = organizationContext({
  workspace_memberships: [
    { workspace_id: WORKSPACE_ID, name: "Platform", role: "owner" },
  ],
})

const policyPath = `${API_ROOT}/workspaces/${WORKSPACE_ID}/code-execution-policy`

function api(policy: ReturnType<typeof workspaceCodeExecutionPolicy>) {
  return {
    [`${API_ROOT}/organizations/me`]: CONTEXT,
    [`${API_ROOT}/workspaces`]: [
      workspace({ id: WORKSPACE_ID, name: "Platform" }),
    ],
    [policyPath]: policy,
  }
}

const meta = {
  title: "Dashboard/Tools/WorkspaceCodeExecutionPolicyCard",
  component: WorkspaceCodeExecutionPolicyCard,
  parameters: {
    api: api(workspaceCodeExecutionPolicy({ workspace_id: WORKSPACE_ID })),
    layout: "padded",
  },
} satisfies Meta<typeof WorkspaceCodeExecutionPolicyCard>

export default meta

type Story = StoryObj<typeof meta>

/** No policy, which is what a workspace has until somebody sets one: it reads as allowed. */
export const Unconfigured: Story = {
  render: (args) => (
    <div className="w-[44rem]">
      <WorkspaceCodeExecutionPolicyCard {...args} />
    </div>
  ),
}

/** A policy naming a tool the sandbox no longer serves, so it offers a reset. */
export const StalePolicy: Story = {
  parameters: {
    api: api(
      workspaceCodeExecutionPolicy({
        workspace_id: WORKSPACE_ID,
        configured: true,
        enabled: true,
        tools: ["bash_code_execution"],
        created_at: "2026-06-01T00:00:00+00:00",
        updated_at: "2026-08-01T00:00:00+00:00",
      }),
    ),
  },
  render: (args) => (
    <div className="w-[44rem]">
      <WorkspaceCodeExecutionPolicyCard {...args} />
    </div>
  ),
}

/** Explicitly off, which is a decision rather than an absence. */
export const ExplicitlyDisabled: Story = {
  parameters: {
    api: api(
      workspaceCodeExecutionPolicy({
        workspace_id: WORKSPACE_ID,
        configured: true,
        enabled: false,
        created_at: "2026-06-01T00:00:00+00:00",
      }),
    ),
  },
  render: (args) => (
    <div className="w-[44rem]">
      <WorkspaceCodeExecutionPolicyCard {...args} />
    </div>
  ),
}

/**
 * No sandbox backend is configured on the deployment, so the stance is moot: there
 * is nothing to execute code in whatever the workspace asks for.
 */
export const NoSandboxConfigured: Story = {
  parameters: {
    api: api(
      workspaceCodeExecutionPolicy({
        workspace_id: WORKSPACE_ID,
        sandbox_configured: false,
      }),
    ),
  },
  render: (args) => (
    <div className="w-[44rem]">
      <WorkspaceCodeExecutionPolicyCard {...args} />
    </div>
  ),
}
