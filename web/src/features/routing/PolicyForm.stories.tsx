import type { Meta, StoryObj } from "@storybook/react-vite"

import type { PolicySpec } from "@/client"
import { API_ROOT } from "@/shared/api/client"
import {
  catalogModelSummary,
  catalogResponse,
  organizationMember,
  user,
} from "@/tests/fixtures"

import { PolicyForm } from "./PolicyForm"

/**
 * The routing form, in the dialog every create and edit in the dashboard opens
 * in.
 *
 * It owns queries of its own (the model catalog behind every picker, and the
 * tool settings that decide whether guardrails can be offered at all), so
 * these stories declare a stub gateway through `parameters.api` rather than
 * mocking the hooks. The catalog stub answers every search with the same
 * models, since the matching is the server's.
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

const GUARDRAILS_ON = {
  fields: [{ key: "guardrails_url", value: "https://guardrails.internal" }],
}

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
  [`${API_ROOT}/tool-settings`]: GUARDRAILS_ON,
  // The scope picker's two reads, which the form makes while it is open, so
  // without these the mock answers its deliberate 501 rather than the
  // network's 404.
  [`${API_ROOT}/users`]: OWNERS,
  [`${API_ROOT}/organizations/me/members`]: ROSTER,
}

const meta = {
  title: "Features/Routing/PolicyForm",
  component: PolicyForm,
  parameters: { layout: "fullscreen", api: API },
  args: {
    existing: null,
    // The operator's surface, which is what the user-scope picker needs.
    isDeploymentWide: true,
    workspaceId: null,
    onClose: () => {},
  },
} satisfies Meta<typeof PolicyForm>

export default meta

type Story = StoryObj<typeof meta>

const stored = {
  kind: "policy" as const,
  user_id: null,
  workspace_id: null,
  source: "stored",
  is_dynamic: false,
  created_at: "2026-09-10T12:00:00Z",
  updated_at: "2026-09-10T12:00:00Z",
}

/** A new policy: a name, a main model and its backups, with the rest folded. */
export const New: Story = {}

/** The common case, edited: one model with two backups behind it, in order. */
export const Fallback: Story = {
  args: {
    existing: {
      ...stored,
      name: "fast",
      spec: {
        select: [{ default: "bedrock:anthropic.claude-sonnet-4-5-v1:0" }],
        on_failure: ["openai:gpt-4o", "anthropic:claude-3-5-haiku-latest"],
      } satisfies PolicySpec,
    },
  },
}

/**
 * The longest policy this form can author, which opens with Advanced showing:
 * a fallback chain, a condition tier, a guardrail, and the weighted split that
 * replaces the single target.
 *
 * This is the scrolling body in its real form rather than a demo one. The header
 * and the footer stay put, the body scrolls between them, and the header gains
 * its rule once content is under it.
 */
export const LongestPolicy: Story = {
  args: {
    existing: {
      ...stored,
      name: "balanced",
      // The spec is a `select` list, read in order, plus a shared failure
      // chain and guardrails: a condition tier first, then the weighted split
      // that serves everything else.
      spec: {
        select: [
          {
            // The one condition the form models: tier down once the
            // owner's budget is most of the way spent.
            when: { budget_used_pct: { gte: 80 } },
            target: "anthropic:claude-3-5-haiku-latest",
          },
          {
            router: "weighted",
            candidates: ["openai:gpt-4o", "anthropic:claude-sonnet-4-5"],
            weights: { "openai:gpt-4o": 70, "anthropic:claude-sonnet-4-5": 30 },
          },
          // The fallthrough is its own entry and it is last, which is the shape
          // the form writes: an entry after the default can never be reached.
          { default: "openai:gpt-4o" },
        ],
        on_failure: ["anthropic:claude-sonnet-4-5", "openai:gpt-4o-mini"],
        guardrails: [
          { profile: "pii", mode: "block", on_unavailable: "block" },
        ],
      } satisfies PolicySpec,
    },
  },
}

/**
 * No guardrails service configured, so the affordance under Advanced says why
 * rather than offering a control that would have nothing to call.
 */
export const WithoutGuardrailsService: Story = {
  parameters: {
    api: { ...API, [`${API_ROOT}/tool-settings`]: { fields: [] } },
  },
}

/**
 * The longest policy again, with a catalog that could not be read.
 *
 * The pickers still take a typed selector, and the advanced rows report the
 * failure under each row rather than under the picker inside one, which is
 * what keeps every control in the row on one line; `web/design/forms.md`
 * ("Control rows") says why a field cannot hold it.
 */
export const CatalogUnavailable: Story = {
  ...LongestPolicy,
  parameters: {
    api: { ...API, [`${API_ROOT}/catalog/models`]: { $status: 503 } },
  },
}
