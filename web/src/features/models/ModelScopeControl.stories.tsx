import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"

import { accessLabel, ModelScopeControl } from "./ModelScopeControl"

/**
 * The allow-list editor: either any model, or an explicit list.
 *
 * One control serves two callers, a user's default scope and a key's override,
 * which is why the copy is configurable and the value is `string[] | null`, where
 * `null` means "any" rather than "none".
 *
 * It reads three queries (providers, discoverable models, aliases) to suggest
 * targets, so these stories declare all three. `onChange` reports validity
 * alongside the value, so the page can hold its submit while an entry is
 * incomplete.
 */
// A `DiscoverableModel` is `{ id, key }`, not a string: `id` is the bare model
// id the provider reports and `key` is the `instance:model` selector. Same
// correction ModelComboBox's stories needed, and for the same reason: a plain
// string here reaches `model.key.toLowerCase()` as undefined.
const DISCOVERABLE = {
  providers: [
    {
      provider: "openai",
      models: ["gpt-4o", "gpt-4o-mini", "o3-mini"].map((id) => ({
        id,
        key: `openai:${id}`,
      })),
      ok: true,
      discovery_unsupported: false,
      checked_at: "2026-08-25T12:00:00Z",
    },
    {
      provider: "anthropic",
      models: ["claude-opus-4", "claude-haiku-4-5"].map((id) => ({
        id,
        key: `anthropic:${id}`,
      })),
      ok: true,
      discovery_unsupported: false,
      checked_at: "2026-08-25T12:00:00Z",
    },
  ],
}

const PROVIDERS = {
  providers: [
    {
      provider_type: "openai",
      instance: "openai",
      name: "OpenAI",
      capabilities: ["chat"],
    },
    {
      provider_type: "anthropic",
      instance: "anthropic",
      name: "Anthropic",
      capabilities: ["chat"],
    },
  ],
}

const ALIASES = [
  { name: "fast", target: "openai:gpt-4o-mini", source: "config" },
  { name: "smart", target: "anthropic:claude-opus-4", source: "config" },
]

const API = {
  "/v1/models/discoverable": DISCOVERABLE,
  "/v1/providers": PROVIDERS,
  "/v1/aliases": ALIASES,
}

const meta = {
  title: "Dashboard/Models/ModelScopeControl",
  component: ModelScopeControl,
  args: { initial: null, onChange: () => {} },
  parameters: { api: API, layout: "padded" },
} satisfies Meta<typeof ModelScopeControl>

export default meta

type Story = StoryObj<typeof meta>

/** `null` is "any model", which is the default a new user gets. */
export const AnyModel: Story = {
  render: (args) => (
    <div className="w-[36rem]">
      <ModelScopeControl {...args} />
    </div>
  ),
}

/** An explicit list, which is what an override looks like. */
export const ExplicitList: Story = {
  args: { initial: ["openai:gpt-4o-mini", "anthropic:claude-haiku-4-5"] },
  render: (args) => (
    <div className="w-[36rem]">
      <ModelScopeControl {...args} />
    </div>
  ),
}

/**
 * An empty list is not the same as `null`: it allows nothing at all, which the
 * control has to be able to express so a key can be scoped shut.
 */
export const EmptyList: Story = {
  args: { initial: [] },
  render: (args) => (
    <div className="w-[36rem]">
      <ModelScopeControl {...args} />
    </div>
  ),
}

/** The key-override caller, with its own copy. */
export const AsKeyOverride: Story = {
  args: {
    initial: ["openai:gpt-4o-mini"],
    title: "Model access for this key",
    description:
      'Narrower than the owner\'s default. Leave on "inherit" to follow the user.',
    anyLabel: "Inherit from the user",
  },
  render: (args) => (
    <div className="w-[36rem]">
      <ModelScopeControl {...args} />
    </div>
  ),
}

/**
 * The validity half of `onChange`, which is the reason it reports two values: an
 * incomplete row is not a value the page may submit.
 */
export const ReportsValidity: Story = {
  render: (args) => {
    const [state, setState] = useState<{
      value: string[] | null
      valid: boolean
    }>({ value: ["openai:gpt-4o-mini"], valid: true })
    return (
      <div className="flex w-[36rem] flex-col gap-3">
        <ModelScopeControl
          {...args}
          initial={["openai:gpt-4o-mini"]}
          onChange={(value, valid) => setState({ value, valid })}
        />
        <p className="text-caption">
          valid: {String(state.valid)}, value:{" "}
          {state.value === null ? "any" : JSON.stringify(state.value)}
        </p>
      </div>
    )
  },
}

/**
 * `accessLabel` is the read-only counterpart, used in a table cell where the whole
 * editor would not fit. Its `tone` is what makes "no models" read as a problem
 * rather than as a setting.
 */
export const AccessLabels: Story = {
  render: () => (
    <div className="flex w-[24rem] flex-col gap-2">
      {[null, [], ["openai:gpt-4o-mini"], ["a", "b", "c", "d"]].map(
        (allowed, index) => {
          const label = accessLabel(allowed)
          const tone =
            label.tone === "danger"
              ? "text-danger"
              : label.tone === "muted"
                ? "text-muted"
                : "text-foreground"
          return (
            <div key={index} className="flex items-center justify-between">
              <span className="font-mono text-caption">
                {allowed === null ? "null" : JSON.stringify(allowed)}
              </span>
              <span className={tone}>{label.text}</span>
            </div>
          )
        },
      )}
    </div>
  ),
}
