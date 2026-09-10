import type { Meta, StoryObj } from "@storybook/react-vite"

import { Chip, type ChipTone } from "./Chip"

/**
 * A short, closed-vocabulary label on a fill: a state, a scope, a plan.
 *
 * Its own element rather than HeroUI's `Chip`, because two of HeroUI's own
 * tone-to-ink pairings measure under AA against these tokens. The accent is the
 * one to know: `--color-primary` on `--color-primary-subtle` is 3.8:1, so the
 * `accent` tone below wears `text-primary-subtle-foreground` instead. Switch the
 * **Theme** toolbar to check both artboards.
 *
 * A chip is not a status mark. Reach for one when the label *is* the value
 * ("read-only", "Free plan"); reach for `Dot` or `SeverityMark` for a judgement
 * about health, because those carry a word beside a square and survive a
 * red-green deficiency.
 */
const meta = {
  title: "Design system/Indicators/Chip",
  component: Chip,
  args: { children: "read-only" },
  parameters: { layout: "padded" },
} satisfies Meta<typeof Chip>

export default meta

type Story = StoryObj<typeof meta>

const TONES: ChipTone[] = [
  "neutral",
  "accent",
  "success",
  "warning",
  "danger",
  "info",
]

export const Default: Story = {}

/** Every tone, each with the one ink that clears AA on its own fill. */
export const Tones: Story = {
  render: () => (
    <div className="flex flex-wrap items-center gap-2">
      {TONES.map((tone) => (
        <Chip key={tone} tone={tone}>
          {tone}
        </Chip>
      ))}
    </div>
  ),
}

export const TonesDark: Story = {
  render: () => (
    <div className="flex flex-wrap items-center gap-2">
      {TONES.map((tone) => (
        <Chip key={tone} tone={tone}>
          {tone}
        </Chip>
      ))}
    </div>
  ),
  globals: { theme: "dark" },
}

/** Two sizes. Both keep the caption size; only the padding changes. */
export const Sizes: Story = {
  render: () => (
    <div className="flex items-center gap-2">
      <Chip size="sm">small</Chip>
      <Chip size="md">medium</Chip>
    </div>
  ),
}

/** What they look like in the rows they actually ship in. */
export const InContext: Story = {
  render: () => (
    <div className="flex w-96 flex-col gap-2">
      <div className="flex items-center justify-between gap-2 border-b border-border py-2">
        <span className="font-mono text-mono-caption">gpt-4o-mini</span>
        <Chip tone="success">priced</Chip>
      </div>
      <div className="flex items-center justify-between gap-2 border-b border-border py-2">
        <span className="font-mono text-mono-caption">claude-opus-4</span>
        <Chip tone="warning">no price</Chip>
      </div>
      <div className="flex items-center justify-between gap-2 border-b border-border py-2">
        <span className="font-mono text-mono-caption">llama-3.3-70b</span>
        <Chip tone="neutral">every workspace</Chip>
      </div>
    </div>
  ),
}
