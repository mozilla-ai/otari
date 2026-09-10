import type { Meta, StoryObj } from "@storybook/react-vite"

import {
  activationAttempt,
  organizationContext,
  workspaceActivation,
} from "@/tests/fixtures"

import { SetupGuideCard } from "./SetupGuideCard"

/**
 * The first-request guide: mint a setup key, copy two curl calls, watch for the
 * request to land.
 *
 * `canServeRequests` is a prop rather than a query of this component's own, and
 * that is load-bearing: the Overview page already owns that query, and a second
 * observer in here would re-trigger it on mount, flip the page back to its
 * loading branch, and unmount the observer that asked. One query, one owner.
 *
 * It says whether a request from this caller could succeed, which is not the same
 * question as "a provider exists": an operator's Overview reads it off
 * `/v1/providers`, which refuses a tenant, and an organization's off the model
 * catalog, which lists the selectors that caller may name.
 *
 * It polls `/v1/workspaces/{id}/activation` for the payoff, so the states below are
 * that endpoint's: waiting, succeeded, failed. The workspace comes from
 * `SelectedWorkspaceProvider`, which is why every story mocks
 * `/v1/organizations/me`.
 */
const WORKSPACE_ID = "44444444-4444-4444-4444-444444444444"

const CONTEXT = organizationContext({
  workspace_memberships: [
    { workspace_id: WORKSPACE_ID, name: "Platform", role: "owner" },
  ],
})

const activationPath = `/v1/workspaces/${WORKSPACE_ID}/activation`

function api(activation: ReturnType<typeof workspaceActivation>) {
  return {
    "/v1/organizations/me": CONTEXT,
    [activationPath]: activation,
    "/v1/models": { models: [], total: 0 },
  }
}

const meta = {
  title: "Dashboard/Onboarding/SetupGuideCard",
  component: SetupGuideCard,
  args: { canServeRequests: true },
  parameters: { api: api(workspaceActivation()), layout: "padded" },
} satisfies Meta<typeof SetupGuideCard>

export default meta

type Story = StoryObj<typeof meta>

/** Waiting for the first request, which is the state the guide exists for. */
export const Waiting: Story = {
  render: (args) => (
    <div className="w-[46rem]">
      <SetupGuideCard {...args} />
    </div>
  ),
}

/**
 * No provider configured yet, and the guide renders **nothing at all**.
 *
 * That is deliberate rather than a gap: `canServeRequests: false` disables the
 * activation query, so there is no data and the card returns null. With nothing
 * to route to, a setup key would be handing out a credential for a call that
 * cannot succeed, so the Overview's own getting-started panel is the right guide
 * instead.
 */
export const NoProviders: Story = {
  args: { canServeRequests: false },
  render: (args) => (
    <div className="flex w-[46rem] flex-col gap-2">
      <SetupGuideCard {...args} />
      <p className="text-caption">Nothing above this line, deliberately.</p>
    </div>
  ),
}

/** The payoff: a request landed, with the attempt that proves it. */
export const Activated: Story = {
  parameters: {
    api: api(
      workspaceActivation({
        status: "activated",
        activation_attempt: activationAttempt(),
        latest_attempt: activationAttempt(),
      }),
    ),
  },
  render: (args) => (
    <div className="w-[46rem]">
      <SetupGuideCard {...args} />
    </div>
  ),
}

/**
 * A request arrived and failed. The guide reports the category rather than a raw
 * provider error, because the operator's next move depends on which of the handful
 * of causes it was.
 */
export const AttemptFailed: Story = {
  parameters: {
    api: api(
      workspaceActivation({
        status: "waiting",
        latest_attempt: activationAttempt({
          status: "failed",
          error_category: "configuration",
          cost_usd: 0,
          latency_ms: 118,
        }),
      }),
    ),
  },
  render: (args) => (
    <div className="w-[46rem]">
      <SetupGuideCard {...args} />
    </div>
  ),
}

/** Dismissed, so the Overview page stops offering it. */
export const Dismissed: Story = {
  parameters: {
    api: api(workspaceActivation({ dismissed: true })),
  },
  render: (args) => (
    <div className="w-[46rem]">
      <SetupGuideCard {...args} />
    </div>
  ),
}
