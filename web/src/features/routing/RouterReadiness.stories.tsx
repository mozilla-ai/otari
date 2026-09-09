import type { Meta, StoryObj } from "@storybook/react-vite"

import { user } from "@/tests/fixtures"

import { RouterReadiness } from "./RouterReadiness"

/**
 * How warm a learned routing policy is, for one user at a time.
 *
 * The panel exists because a learned policy is only as good as the records behind
 * it: a cold pool routes to the default target, which looks like the policy being
 * ignored. It reads `useUsers` and `useRouterStatus`, and the status query is
 * per-user (`enabled` only once a user is picked), which is why the picker is part
 * of the panel rather than the page.
 */
const USERS = [
  user({ user_id: "ops@example.com" }),
  user({ user_id: "dev@example.com" }),
]

function status(overrides: Record<string, unknown> = {}) {
  return {
    user_id: "ops@example.com",
    embedding_model: "openai:text-embedding-3-small",
    granularity: "task",
    alpha: 0.6,
    k: 8,
    confidence_floor: 0.35,
    seed_count: 24,
    default_pool: { records: 412, warm: true },
    tasks: [
      { task_id: "summarize", records: 218, warm: true },
      { task_id: "classify", records: 96, warm: true },
      { task_id: "extract", records: 11, warm: false },
    ],
    policies: [
      {
        name: "cost-aware",
        backend: "learned",
        candidates: ["openai:gpt-4o-mini", "anthropic:claude-haiku-4-5"],
        default_target: "openai:gpt-4o-mini",
      },
    ],
    ...overrides,
  }
}

const meta = {
  title: "Dashboard/Routing/RouterReadiness",
  component: RouterReadiness,
  args: {
    policyName: "cost-aware",
    candidates: ["openai:gpt-4o-mini", "anthropic:claude-haiku-4-5"],
    defaultTarget: "openai:gpt-4o-mini",
    backend: "learned",
    scopedUserId: "ops@example.com",
    onClose: () => {},
  },
  parameters: {
    api: {
      "/v1/users": USERS,
      "/v1/routing/status": status(),
    },
    layout: "padded",
  },
} satisfies Meta<typeof RouterReadiness>

export default meta

type Story = StoryObj<typeof meta>

/** A warm pool with a few warm tasks, which is a policy doing its job. */
export const Warm: Story = {
  render: (args) => (
    <div className="w-[44rem]">
      <RouterReadiness {...args} />
    </div>
  ),
}

/**
 * Cold. This is the state the panel is really for: every request falls through to
 * the default target, which from the outside is indistinguishable from the policy
 * being misconfigured.
 */
export const Cold: Story = {
  parameters: {
    api: {
      "/v1/users": USERS,
      "/v1/routing/status": status({
        seed_count: 0,
        default_pool: { records: 0, warm: false },
        tasks: [],
      }),
    },
  },
  render: (args) => (
    <div className="w-[44rem]">
      <RouterReadiness {...args} />
    </div>
  ),
}

/**
 * No user picked yet. The status query is `enabled` only once one is, so the panel
 * asks rather than showing a global number that would mean nothing.
 */
export const NoUserScoped: Story = {
  args: { scopedUserId: null },
  render: (args) => (
    <div className="w-[44rem]">
      <RouterReadiness {...args} />
    </div>
  ),
}

/** A policy with many candidates, which is where the list has to stay readable. */
export const ManyCandidates: Story = {
  args: {
    candidates: [
      "openai:gpt-4o",
      "openai:gpt-4o-mini",
      "openai:o3-mini",
      "anthropic:claude-opus-4",
      "anthropic:claude-haiku-4-5",
      "mistral:mistral-large",
    ],
  },
  render: (args) => (
    <div className="w-[44rem]">
      <RouterReadiness {...args} />
    </div>
  ),
}

/** A non-learned backend, where warmth is not the question at all. */
export const StaticBackend: Story = {
  args: { backend: "weighted", policyName: "split-traffic" },
  render: (args) => (
    <div className="w-[44rem]">
      <RouterReadiness {...args} />
    </div>
  ),
}
