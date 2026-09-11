import type { Meta, StoryObj } from "@storybook/react-vite"

import type { ReactNode } from "react"
import type { CardRatio } from "./ShareCard"
import { ShareCard } from "./ShareCard"
import type { CardModel, CardStat } from "./shareCardData"

/**
 * The usage share image: a standalone artifact an operator posts.
 *
 * It is the one place in `src/` allowed to name colors literally, and
 * `src/styles/foundation.test.ts` exempts exactly this file. The card is
 * rasterized through an `<img>`-loaded SVG document, where a custom property does
 * not resolve, so its palette has to be self-contained, which is why `theme` is
 * a prop here rather than being inherited from the document. Do not "fix" it to
 * use tokens; the export would come out unstyled.
 *
 * That also makes it the cleanest thing in the tree to put in a catalog: pure
 * DOM, no context, no fetch, and every pixel decided by props.
 *
 * It renders at its true export size (1080px square, 1200x630 landscape) and is
 * scaled down here so both fit on screen.
 */
const HERO: CardStat = { id: "requests", label: "Requests", value: "18,402" }

const STATS: CardStat[] = [
  { id: "tokens", label: "Tokens", value: "4.1M" },
  { id: "cost", label: "Spend", value: "$412.90", caveated: true },
  { id: "latency", label: "p50 latency", value: "740ms" },
]

const MODELS: CardModel[] = [
  {
    key: "anthropic:claude-haiku-4-5",
    label: "claude-haiku-4-5",
    tokens: 2_400_000,
    cost: 118.2,
    requests: 11_204,
    isOther: false,
  },
  {
    key: "openai:gpt-4o-mini",
    label: "gpt-4o-mini",
    tokens: 1_180_000,
    cost: 84.6,
    requests: 5_812,
    isOther: false,
  },
  {
    key: undefined,
    label: "other",
    tokens: 520_000,
    cost: 210.1,
    requests: 1_386,
    isOther: true,
  },
]

const meta = {
  title: "Dashboard/Usage/ShareCard",
  component: ShareCard,
  args: {
    ratio: "square",
    theme: "dark",
    title: "Otari usage",
    scope: "Last 30 days, all workspaces",
    hero: HERO,
    models: MODELS,
    stats: STATS,
    unpricedRequests: 1_386,
  },
  parameters: { layout: "padded" },
} satisfies Meta<typeof ShareCard>

export default meta

type Story = StoryObj<typeof meta>

/**
 * Scale the true export size down to something a screen can hold.
 *
 * Fixed arbitrary values rather than an inline `transform` style, because the
 * dashboard styles from classes and the sizes are known: 1080px is 67.5rem and
 * 1200x630 is 75rem x 39.375rem, so a 0.4 scale lands on 27rem and 30rem x
 * 15.75rem. `origin-top-left` is what makes the scaled box start at the corner
 * instead of shrinking toward its middle and leaving the crop off-centre.
 */
function Scaled({
  ratio,
  children,
}: {
  ratio: CardRatio
  children: ReactNode
}) {
  const frame =
    ratio === "square" ? "h-[27rem] w-[27rem]" : "h-[15.75rem] w-[30rem]"
  const inner =
    ratio === "square" ? "h-[67.5rem] w-[67.5rem]" : "h-[39.375rem] w-[75rem]"
  return (
    <div className={`overflow-hidden rounded-lg border border-border ${frame}`}>
      <div className={`origin-top-left scale-[0.4] ${inner}`}>{children}</div>
    </div>
  )
}

/** All four combinations, which is the whole matrix the share dialog can export. */
export const AllRatiosAndThemes: Story = {
  render: (args) => (
    <div className="flex flex-wrap items-start gap-6">
      {(["square", "landscape"] as const).map((ratio) =>
        (["dark", "light"] as const).map((theme) => (
          <div key={`${ratio}-${theme}`} className="flex flex-col gap-2">
            <span className="text-overline">
              {ratio} / {theme}
            </span>
            <Scaled ratio={ratio}>
              <ShareCard {...args} ratio={ratio} theme={theme} />
            </Scaled>
          </div>
        )),
      )}
    </div>
  ),
}

export const SquareDark: Story = {
  render: (args) => (
    <Scaled ratio="square">
      <ShareCard {...args} />
    </Scaled>
  ),
}

export const LandscapeLight: Story = {
  args: { ratio: "landscape", theme: "light" },
  render: (args) => (
    <Scaled ratio="landscape">
      <ShareCard {...args} />
    </Scaled>
  ),
}

/**
 * No caveated stat, so the asterisk and its legend are both absent. The legend
 * exists because the card is posted as a standalone file, where a mark whose
 * meaning lives only in the docs is unreadable.
 */
export const NoUnpricedCaveat: Story = {
  args: {
    stats: STATS.filter((stat) => !stat.caveated),
    unpricedRequests: 0,
  },
  render: (args) => (
    <Scaled ratio="square">
      <ShareCard {...args} />
    </Scaled>
  ),
}

/**
 * A gateway serving only self-hosted models: dollars hidden, so cost is omitted
 * rather than published as a proud "$0.00".
 */
export const NoDollars: Story = {
  args: {
    hero: { id: "tokens", label: "Tokens", value: "4.1M" },
    stats: [
      { id: "requests", label: "Requests", value: "18,402" },
      { id: "latency", label: "p50 latency", value: "740ms" },
      { id: "cacheHitRate", label: "Cache hits", value: "38%" },
    ],
    unpricedRequests: 0,
  },
  render: (args) => (
    <Scaled ratio="square">
      <ShareCard {...args} />
    </Scaled>
  ),
}

/**
 * A long, fully qualified model id. Ids are distinguished by their tails, so the
 * card middle-truncates rather than cutting the identifying part off the end.
 */
export const LongModelNames: Story = {
  args: {
    models: [
      {
        key: "otari.ai:fireworks/accounts/deepseek-v4-flash-preview",
        label: "fireworks/accounts/deepseek-v4-flash-preview",
        tokens: 3_100_000,
        cost: 96.4,
        requests: 8_912,
        isOther: false,
      },
      {
        key: "anthropic:claude-sonnet-4-5-20260514",
        label: "claude-sonnet-4-5-20260514",
        tokens: 1_900_000,
        cost: 302.8,
        requests: 4_180,
        isOther: false,
      },
    ],
  },
  render: (args) => (
    <Scaled ratio="square">
      <ShareCard {...args} />
    </Scaled>
  ),
}

/** A single model and a single stat, which a fresh gateway's first week looks like. */
export const Sparse: Story = {
  args: {
    scope: "Last 24 hours",
    hero: { id: "requests", label: "Requests", value: "87" },
    models: [MODELS[0]],
    stats: [{ id: "tokens", label: "Tokens", value: "12.4K" }],
    unpricedRequests: 0,
  },
  render: (args) => (
    <Scaled ratio="square">
      <ShareCard {...args} />
    </Scaled>
  ),
}
