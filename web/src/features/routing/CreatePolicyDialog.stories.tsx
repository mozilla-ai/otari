import type { Meta, StoryObj } from "@storybook/react-vite"

import { API_ROOT } from "@/shared/api/client"
import {
  catalogModelSummary,
  catalogResponse,
  organizationMember,
  user,
} from "@/tests/fixtures"

import { CreatePolicyDialog } from "./CreatePolicyDialog"

/**
 * The dialog a new routing policy is created in: what it is for first, then a
 * short form for that alone.
 *
 * It reads the model catalog behind every picker and, for an operator, the
 * user roster, so these stories declare a stub gateway through
 * `parameters.api` rather than mocking the hooks.
 */
const CATALOG = catalogResponse([
  catalogModelSummary({
    id: "openai/gpt-4o",
    name: "GPT-4o",
    selectors: ["azure:gpt-4o", "openai:gpt-4o"],
    providers: ["azure", "openai"],
    resolves_to: "openai:gpt-4o",
  }),
  catalogModelSummary({
    id: "openai/gpt-4o-mini",
    name: "GPT-4o mini",
    selectors: ["openai:gpt-4o-mini"],
    providers: ["openai"],
    resolves_to: "openai:gpt-4o-mini",
  }),
  catalogModelSummary({
    id: "anthropic/claude-sonnet-4.5",
    name: "Claude Sonnet 4.5",
    selectors: [
      "anthropic:claude-sonnet-4-5",
      "bedrock:anthropic.claude-sonnet-4-5-v1:0",
    ],
    providers: ["anthropic", "bedrock"],
    resolves_to: "anthropic:claude-sonnet-4-5",
  }),
  catalogModelSummary({
    id: "anthropic/claude-3.5-haiku",
    name: "Claude 3.5 Haiku",
    selectors: ["anthropic:claude-3-5-haiku-latest"],
    providers: ["anthropic"],
    resolves_to: "anthropic:claude-3-5-haiku-latest",
  }),
])

/** Who the scope picker can name, and what the roster calls them.
 *
 *  Both halves of the labeling, so the scoped tab shows the real thing: one
 *  owner id the organization has a person behind, and one it does not.
 */
const OWNERS = [user({ user_id: "u-ana" }), user({ user_id: "release-bot" })]

const ROSTER = {
  data: [
    organizationMember({
      user_id: "u-ana",
      attribution_user_id: "u-ana",
      full_name: "Ana Ruiz",
      role: "member",
    }),
  ],
  count: 1,
}

const API = {
  [`${API_ROOT}/catalog/models`]: CATALOG,
  // The scope picker's two reads, made while the dialog is open.
  [`${API_ROOT}/users`]: OWNERS,
  [`${API_ROOT}/organizations/me/members`]: ROSTER,
}

const meta = {
  title: "Features/Routing/CreatePolicyDialog",
  component: CreatePolicyDialog,
  parameters: { layout: "fullscreen", api: API },
  args: {
    isOpen: true,
    // The operator's surface, which is what the user-scope picker needs.
    isDeploymentWide: true,
    workspaceId: null,
    onClose: () => {},
  },
} satisfies Meta<typeof CreatePolicyDialog>

export default meta

type Story = StoryObj<typeof meta>

/** The choice every new policy starts from, one line per kind. */
export const Choice: Story = {}

/** The common case: a name, a main model and its backups. */
export const Fallback: Story = {
  args: { startAt: "fallback", initialTarget: "openai:gpt-4o" },
}

export const SplitTraffic: Story = { args: { startAt: "weighted" } }

export const SmartRouter: Story = { args: { startAt: "knn" } }

export const SpillOver: Story = { args: { startAt: "priority" } }

export const TierDown: Story = { args: { startAt: "tierdown" } }

/** An organization admin, whose policy applies to the whole workspace. */
export const TenantScoped: Story = {
  args: {
    startAt: "fallback",
    isDeploymentWide: false,
    workspaceId: "44444444-4444-4444-4444-444444444444",
  },
}
