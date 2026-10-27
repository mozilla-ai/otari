import type { Meta, StoryObj } from "@storybook/react-vite"
import type { ReactNode } from "react"

import { ContextMapView } from "./ContextMapView"
import {
  buildContextMap,
  type ContextMapInput,
  type ContextWeights,
} from "./contextMap"

/**
 * What filled the model's context window on one request, as blocks sized by
 * tokens, with the unused part of the window hatched as Free.
 *
 * The input total is the provider's own count. A request Otari served itself
 * also kept the size of each part in characters (never text), which the map
 * scales to that total and marks **Estimated**. Any other model request, such as
 * one a coding agent exported over OTLP, has only the cache split, which is
 * exact. Hover, tab to or tap a block for its count and share.
 */
const meta = {
  title: "Dashboard/Traces/ContextMap",
  component: ContextMapView,
  parameters: { layout: "padded" },
} satisfies Meta<typeof ContextMapView>

export default meta

type Story = StoryObj<typeof meta>

// A coding-agent request: a long system prompt, a big toolbox, a long history
// and a lot of tool output.
const WEIGHTS: ContextWeights = {
  system: 21_000,
  tools: 59_000,
  user: 12_000,
  assistant: 36_000,
  toolResults: 70_400,
  files: 4_000,
  toolCount: 12,
  messageCount: 41,
}

const ESTIMATED: ContextMapInput = {
  inputTokens: 39_600,
  cacheReadTokens: 31_200,
  cacheWriteTokens: 2_400,
  windowTokens: 200_000,
  weights: WEIGHTS,
}

const CACHE_ONLY: ContextMapInput = {
  inputTokens: 48_210,
  cacheReadTokens: 41_900,
  cacheWriteTokens: 6_010,
  windowTokens: 200_000,
}

function Frame({ children }: { children: ReactNode }) {
  return <div className="w-full max-w-3xl">{children}</div>
}

/** A request Otari served: the whole composition, estimated, in a 200k window. */
export const EstimatedComposition: Story = {
  args: { map: buildContextMap(ESTIMATED) },
  render: (args) => (
    <Frame>
      <ContextMapView {...args} />
    </Frame>
  ),
}

/** A model request from an OTLP exporter (Claude Code, Codex): only the cache split is known. */
export const CacheSplitOnly: Story = {
  args: { map: buildContextMap(CACHE_ONLY) },
  render: (args) => (
    <Frame>
      <ContextMapView {...args} />
    </Frame>
  ),
}

/** The catalog does not know the model: the used tokens, and no Free block. */
export const WindowUnknown: Story = {
  args: {
    map: buildContextMap({ ...ESTIMATED, windowTokens: undefined }),
  },
  render: (args) => (
    <Frame>
      <ContextMapView {...args} />
    </Frame>
  ),
}

/** The input is larger than the recorded window (compacted, or a wrong window): capped, no Free. */
export const OverWindow: Story = {
  args: {
    map: buildContextMap({
      ...CACHE_ONLY,
      inputTokens: 231_000,
      windowTokens: 200_000,
    }),
  },
  render: (args) => (
    <Frame>
      <ContextMapView {...args} />
    </Frame>
  ),
}

/** Small parts keep their name for a screen reader and a tooltip, and lose the text they have no room for. */
export const SmallBlocks: Story = {
  args: {
    map: buildContextMap({
      inputTokens: 128_000,
      windowTokens: 1_000_000,
      weights: {
        ...WEIGHTS,
        system: 900,
        files: 0,
        tools: 2_500,
        user: 700,
        assistant: 500,
      },
    }),
  },
  render: (args) => (
    <Frame>
      <ContextMapView {...args} />
    </Frame>
  ),
}

/** A span that recorded no input tokens. */
export const NoInput: Story = {
  args: { map: buildContextMap({ inputTokens: 0 }) },
}

/** Narrow, as in the panel on a phone. */
export const Narrow: Story = {
  args: { map: buildContextMap(ESTIMATED) },
  render: (args) => (
    <div className="w-80">
      <ContextMapView {...args} />
    </div>
  ),
}

/** The normal trace side panel, about 280px: parts too small to be written on their blocks are listed beneath the map. */
export const SidePanel: Story = {
  args: {
    map: buildContextMap(ESTIMATED),
    lead: "From the last of 31 requests.",
  },
  render: (args) => (
    <div className="w-[280px]">
      <ContextMapView {...args} />
    </div>
  ),
}

/** Both themes at once, each scoped by `data-theme`, so one screenshot shows both. */
export const LightAndDark: Story = {
  args: { map: buildContextMap(ESTIMATED) },
  render: (args) => (
    <div className="grid w-full max-w-6xl gap-4 lg:grid-cols-2">
      {(["light", "dark"] as const).map((theme) => (
        <div
          key={theme}
          data-theme={theme}
          className="flex flex-col gap-6 bg-background p-4 text-body"
        >
          <ContextMapView {...args} />
          <ContextMapView map={buildContextMap(CACHE_ONLY)} />
        </div>
      ))}
    </div>
  ),
}
