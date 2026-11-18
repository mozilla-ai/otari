import type { Meta, StoryObj } from "@storybook/react-vite"

import type { AutomaticFallback, FallbackOffering } from "@/client"

import { AutomaticFallbacks } from "./AutomaticFallbacks"

/** The routing page's list of fallbacks a deployment gives without a policy. */
const meta = {
  title: "Routing/AutomaticFallbacks",
  component: AutomaticFallbacks,
} satisfies Meta<typeof AutomaticFallbacks>

export default meta
type Story = StoryObj<typeof meta>

const offering = (
  selector: string,
  input_price: number | null,
): FallbackOffering => {
  const provider = selector.split(":")[0]
  return {
    selector,
    provider,
    provider_type: provider,
    input_price,
    own_tools: [],
    otari_tools: [],
  }
}

const fallback = (
  overrides: Partial<AutomaticFallback> &
    Pick<AutomaticFallback, "id" | "offerings">,
): AutomaticFallback => ({
  name: overrides.id,
  requests: 0,
  fell_back: 0,
  pinned: 0,
  pinned_provider: null,
  pinned_keys: [],
  ...overrides,
})

const FALLBACKS: AutomaticFallback[] = [
  fallback({
    id: "deepseek/deepseek-v4.1-flash-1017",
    offerings: [
      offering("deepinfra:deepseek-ai/DeepSeek-V4.1-Flash-1017", 0.098),
      offering("nebius:deepseek-ai/DeepSeek-V4.1-Flash-1017", 0.3),
      offering("together:deepseek-ai/DeepSeek-V4.1-Flash-1017", 0.3),
    ],
    requests: 6600,
    pinned: 5800,
    pinned_provider: "together",
    pinned_keys: [
      {
        api_key_id: "key-zed",
        key_name: "zed-laptop",
        provider: "together",
        requests: 3300,
      },
      {
        api_key_id: "key-desktop",
        key_name: "ci-agent",
        provider: "together",
        requests: 700,
      },
      {
        api_key_id: "key-mbp",
        key_name: "notebook",
        provider: "deepinfra",
        requests: 292,
      },
    ],
  }),
  fallback({
    id: "anthropic/claude-sonnet-5.5",
    offerings: [
      offering("anthropic:claude-sonnet-5-5", 3),
      offering("bedrock:anthropic.claude-sonnet-5-5-v1:0", 3),
      offering("azureanthropic:claude-sonnet-5-5", 3),
    ],
    requests: 1840,
    fell_back: 42,
  }),
  fallback({
    id: "z-ai/glm-5.3",
    offerings: [
      offering("fireworks:accounts/fireworks/models/glm-5p3", 0.4),
      offering("together:zai-org/GLM-5.3", 0.45),
    ],
    requests: 410,
    pinned: 12,
    pinned_provider: "together",
    pinned_keys: [
      {
        api_key_id: null,
        key_name: null,
        provider: "together",
        requests: 12,
      },
    ],
  }),
  fallback({
    id: "google/gemini-3.8-flash",
    offerings: [offering("gemini:gemini-3.8-flash", 0.15)],
    requests: 260,
    pinned: 260,
    pinned_provider: "gemini",
  }),
]

export const Default: Story = { args: { fallbacks: FALLBACKS } }

export const Empty: Story = { args: { fallbacks: [] } }

export const Loading: Story = { args: { fallbacks: [], isLoading: true } }

export const Failed: Story = {
  args: { fallbacks: [], error: new Error("Could not load the fallbacks.") },
}
