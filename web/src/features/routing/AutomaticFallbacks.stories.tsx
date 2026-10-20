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

const OWN = ["code_execution", "web_fetch", "web_search"]

const offering = (
  selector: string,
  tools: "own" | "otari" | "none" = "otari",
): FallbackOffering => {
  const provider = selector.split(":")[0]
  return {
    selector,
    provider,
    provider_type: provider,
    own_tools: tools === "own" ? OWN : [],
    otari_tools: tools === "otari" ? OWN : [],
  }
}

const fallback = (
  overrides: Partial<AutomaticFallback> &
    Pick<AutomaticFallback, "id" | "name" | "offerings">,
): AutomaticFallback => ({
  requests: 0,
  fell_back: 0,
  pinned: 0,
  pinned_provider: null,
  ...overrides,
})

const FALLBACKS: AutomaticFallback[] = [
  fallback({
    id: "anthropic/claude-sonnet-5.5",
    name: "Claude Sonnet 5.5",
    offerings: [
      offering("anthropic:claude-sonnet-5-5", "own"),
      offering("bedrock:anthropic.claude-sonnet-5-5-v1:0"),
      offering("azureanthropic:claude-sonnet-5-5"),
    ],
    requests: 1840,
    fell_back: 42,
  }),
  fallback({
    id: "anthropic/claude-haiku-5.5",
    name: "Claude Haiku 5.5",
    offerings: [
      offering("anthropic:claude-haiku-5-5", "own"),
      offering("bedrock:anthropic.claude-haiku-5-5-v1:0"),
    ],
    requests: 960,
    fell_back: 17,
  }),
  fallback({
    id: "openai/gpt-6.1-sol",
    name: "GPT-6.1 Sol",
    offerings: [
      offering("openai:gpt-6.1-sol", "own"),
      offering("bedrock:openai.gpt-6.1-sol"),
      offering("azureopenai:gpt-6.1-sol"),
    ],
    requests: 410,
    fell_back: 3,
  }),
  fallback({
    id: "google/gemini-3.8-flash",
    name: "Gemini 3.8 Flash",
    offerings: [offering("gemini:gemini-3.8-flash")],
    requests: 260,
    pinned: 260,
    pinned_provider: "gemini",
  }),
  fallback({
    id: "moonshotai/kimi-k3",
    name: "Kimi K3",
    offerings: [
      offering("fireworks:accounts/fireworks/models/kimi-k3"),
      offering("bedrock:us.moonshotai.kimi-k3"),
    ],
    requests: 90,
    pinned: 90,
    pinned_provider: "bedrock",
  }),
]

export const Default: Story = {
  args: {
    fallbacks: FALLBACKS,
    toolsNote:
      "Tools shown for callers that declare them in the provider's own words and leave the choice to Otari: a provider's own where it runs them, Otari's where it does not.",
  },
}

export const Empty: Story = { args: { fallbacks: [] } }

export const Loading: Story = { args: { fallbacks: [], isLoading: true } }

export const Failed: Story = {
  args: { fallbacks: [], error: new Error("Could not load the fallbacks.") },
}
