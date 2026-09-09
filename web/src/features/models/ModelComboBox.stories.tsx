import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"

import { ModelComboBox } from "./ModelComboBox"

/**
 * A flat, searchable `provider:model` picker over what the providers actually
 * serve.
 *
 * It owns its `useDiscoverableModels` query, so these stories declare a stub
 * gateway through `parameters.api`. Flat rather than grouped by provider on
 * purpose: an operator knows the model name, not which provider is currently
 * routed to it.
 */
// A `DiscoverableModel` is `{ id, key }`, not a string: `id` is the bare model id
// the provider reports and `key` is the `instance:model` selector to send as
// `model`. This builder takes the bare ids and derives both, because the mock
// used to hand the component plain strings and `model.key.toLowerCase()` threw
// on every story whose value was non-empty. That failed only when a query was
// typed, so it surfaced as a different set of stories on each catalog run.
function discoverable(providers: { provider: string; models: string[] }[]) {
  return {
    providers: providers.map((entry) => ({
      provider: entry.provider,
      models: entry.models.map((id) => ({
        id,
        key: `${entry.provider}:${id}`,
      })),
      ok: true,
      discovery_unsupported: false,
      checked_at: "2026-08-25T12:00:00Z",
    })),
  }
}

const CATALOG = discoverable([
  {
    provider: "openai",
    models: ["gpt-4o", "gpt-4o-mini", "o3-mini", "text-embedding-3-small"],
  },
  {
    provider: "anthropic",
    models: ["claude-opus-4", "claude-sonnet-4-5", "claude-haiku-4-5"],
  },
  { provider: "mistral", models: ["mistral-large", "mistral-small"] },
])

const meta = {
  title: "Dashboard/Models/ModelComboBox",
  component: ModelComboBox,
  args: { label: "Model", value: "", onChange: () => {} },
  parameters: { api: { "/v1/models/discoverable": CATALOG } },
} satisfies Meta<typeof ModelComboBox>

export default meta

type Story = StoryObj<typeof meta>

export const Empty: Story = {
  render: (args) => {
    const [value, setValue] = useState("")
    return (
      <div className="w-[24rem]">
        <ModelComboBox {...args} value={value} onChange={setValue} />
      </div>
    )
  },
}

export const WithSelection: Story = {
  render: (args) => {
    const [value, setValue] = useState("anthropic:claude-haiku-4-5")
    return (
      <div className="w-[24rem]">
        <ModelComboBox {...args} value={value} onChange={setValue} />
      </div>
    )
  },
}

export const WithDescription: Story = {
  args: {
    value: "openai:gpt-4o-mini",
    description: "Requests naming this model are routed to it directly.",
    isRequired: true,
  },
  render: (args) => (
    <div className="w-[24rem]">
      <ModelComboBox {...args} />
    </div>
  ),
}

/** Model discovery is off, or no provider is configured: nothing to suggest. */
export const NoModels: Story = {
  parameters: { api: { "/v1/models/discoverable": { providers: [] } } },
  render: (args) => (
    <div className="w-[24rem]">
      <ModelComboBox {...args} placeholder="provider:model" />
    </div>
  ),
}

/**
 * A provider that could not be reached. Its models are simply absent from the
 * list; the picker is a suggestion box, not a health display.
 */
export const ProviderUnreachable: Story = {
  parameters: {
    api: {
      "/v1/models/discoverable": {
        providers: [
          ...CATALOG.providers.slice(0, 1),
          {
            provider: "anthropic",
            models: [],
            ok: false,
            discovery_unsupported: false,
            error: "401 Unauthorized",
            checked_at: "2026-08-25T12:00:00Z",
          },
        ],
      },
    },
  },
  render: (args) => (
    <div className="w-[24rem]">
      <ModelComboBox {...args} />
    </div>
  ),
}

/** A large catalog, which is what an aggregator provider produces. */
export const LargeCatalog: Story = {
  parameters: {
    api: {
      "/v1/models/discoverable": discoverable([
        {
          provider: "openrouter",
          models: Array.from(
            { length: 300 },
            (_, index) => `vendor-${index}/model-${index}`,
          ),
        },
      ]),
    },
  },
  render: (args) => (
    <div className="w-[24rem]">
      <ModelComboBox {...args} />
    </div>
  ),
}
