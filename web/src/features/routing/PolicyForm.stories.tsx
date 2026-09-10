import type { Meta, StoryObj } from "@storybook/react-vite"

import type { PolicySpec } from "@/client"
import { API_ROOT } from "@/shared/api/client"
import { PolicyForm } from "./PolicyForm"

/**
 * The routing form, in the dialog every create and edit in the dashboard opens
 * in.
 *
 * It owns four queries of its own (the model catalog behind both pickers, and
 * the tool settings that decide whether guardrails can be offered at all), so
 * these stories declare a stub gateway through `parameters.api` rather than
 * mocking the hooks.
 */
const CATALOG = {
  providers: [
    {
      provider: "openai",
      models: ["gpt-4o", "gpt-4o-mini", "o3-mini"].map((id) => ({
        id,
        key: `openai:${id}`,
      })),
      ok: true,
      discovery_unsupported: false,
      checked_at: "2026-09-10T12:00:00Z",
    },
    {
      provider: "anthropic",
      models: ["claude-sonnet-4-5", "claude-3-5-haiku-latest"].map((id) => ({
        id,
        key: `anthropic:${id}`,
      })),
      ok: true,
      discovery_unsupported: false,
      checked_at: "2026-09-10T12:00:00Z",
    },
  ],
}

const GUARDRAILS_ON = {
  fields: [{ key: "guardrails_url", value: "https://guardrails.internal" }],
}

const meta = {
  title: "Features/Routing/PolicyForm",
  component: PolicyForm,
  parameters: {
    layout: "fullscreen",
    api: {
      [`${API_ROOT}/models/discoverable`]: CATALOG,
      [`${API_ROOT}/tool-settings`]: GUARDRAILS_ON,
      // `ScopePicker` asks for the roster on every render, so without this the
      // mock answers its deliberate 501 rather than the network's 404.
      [`${API_ROOT}/users`]: [],
    },
  },
  args: {
    existing: null,
    workspaceId: null,
    onClose: () => {},
  },
} satisfies Meta<typeof PolicyForm>

export default meta

type Story = StoryObj<typeof meta>

/** A new policy, which is three fields until something more is asked for. */
export const New: Story = {}

/**
 * The longest policy this form can author, and the reason the routing dialog is
 * `lg` on both of its steps: a fallback chain, a condition tier, a guardrail,
 * and the weighted split that replaces the single target.
 *
 * This is the scrolling body in its real form rather than a demo one. The header
 * and the footer stay put, the body scrolls between them, and the header gains
 * its rule once content is under it.
 */
export const LongestPolicy: Story = {
  args: {
    existing: {
      kind: "policy" as const,
      name: "balanced",
      user_id: null,
      workspace_id: null,
      source: "api",
      is_dynamic: false,
      created_at: "2026-09-10T12:00:00Z",
      updated_at: "2026-09-10T12:00:00Z",
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
 * No guardrails service configured, so the affordance says why rather than
 * offering a control that would have nothing to call.
 */
export const WithoutGuardrailsService: Story = {
  parameters: {
    api: {
      [`${API_ROOT}/models/discoverable`]: CATALOG,
      [`${API_ROOT}/tool-settings`]: { fields: [] },
      [`${API_ROOT}/users`]: [],
    },
  },
}
