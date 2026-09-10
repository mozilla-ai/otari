import type { Meta, StoryObj } from "@storybook/react-vite"

import {
  organizationContext,
  workspace,
  workspaceCodeExecutionPolicy,
} from "@/tests/fixtures"

import { WorkspaceCodeExecutionPolicyCard } from "./WorkspaceCodeExecutionPolicyCard"

/**
 * A workspace's stance on code execution, plus the ceilings it may narrow.
 *
 * The three-position stance is the interesting part: a workspace can be
 * unconfigured (inherit), explicitly on, or explicitly off, and "unconfigured" is
 * not the same as "off", clearing a policy hands the decision back to the
 * deployment default, which is why there is a clear mutation as well as a set one.
 *
 * `MAX_ITERATIONS` (25) and `MAX_EXEC_TIMEOUT_S` (60) are ceilings, not defaults:
 * a workspace may narrow them, never widen them.
 *
 * This card reads the *selected* workspace from context rather than a prop, so
 * these stories mock `/v1/organizations/me`, that is what
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

const policyPath = `/v1/workspaces/${WORKSPACE_ID}/code-execution-policy`

function api(policy: ReturnType<typeof workspaceCodeExecutionPolicy>) {
  return {
    "/v1/organizations/me": CONTEXT,
    "/v1/workspaces": [workspace({ id: WORKSPACE_ID, name: "Platform" })],
    [policyPath]: policy,
  }
}

const meta = {
  title: "Dashboard/Tools/WorkspaceCodeExecutionPolicyCard",
  component: WorkspaceCodeExecutionPolicyCard,
  // `docsHref` is the caller's, the way the page passes it: the card renders the
  // link and does not know how a docs URL is built. The page uses
  // `toolsDocs("per-workspace-code-policy")`.
  args: {
    docsHref: "https://example.com/docs/tools#per-workspace-code-policy",
  },
  parameters: {
    api: api(workspaceCodeExecutionPolicy({ workspace_id: WORKSPACE_ID })),
    layout: "padded",
  },
} satisfies Meta<typeof WorkspaceCodeExecutionPolicyCard>

export default meta

type Story = StoryObj<typeof meta>

/**
 * Unconfigured, which is what a workspace has until somebody sets a policy. The
 * deployment default decides, and this card says so rather than implying "off".
 */
export const Unconfigured: Story = {
  render: (args) => (
    <div className="w-[44rem]">
      <WorkspaceCodeExecutionPolicyCard {...args} />
    </div>
  ),
}

/** Explicitly enabled, with both ceilings narrowed below the maximum. */
export const EnabledWithCeilings: Story = {
  parameters: {
    api: api(
      workspaceCodeExecutionPolicy({
        workspace_id: WORKSPACE_ID,
        configured: true,
        enabled: true,
        max_iterations: 8,
        exec_timeout_s: 20,
        default_purpose_hint: "Data analysis over uploaded CSVs.",
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
