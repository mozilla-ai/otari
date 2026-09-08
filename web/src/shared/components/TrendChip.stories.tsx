import type { Meta, StoryObj } from "@storybook/react-vite"

import type { TrendVariant } from "./metrics/TrendChip"
import { TrendChip } from "./metrics/TrendChip"

const meta = {
  title: "Shared/TrendChip",
  component: TrendChip,
  args: { fraction: 0.033 },
  parameters: { layout: "padded" },
} satisfies Meta<typeof TrendChip>

export default meta

type Story = StoryObj<typeof meta>

const VARIANTS: TrendVariant[] = ["primary", "secondary", "tertiary", "soft"]

/**
 * The four fill treatments against the three directions, which is the matrix the
 * design canvas draws. Nothing here names a color: every fill and ink resolves
 * through the tokens, so the **Theme** toolbar switches the whole grid between
 * the light and dark artboards.
 *
 * The flat chip is the same in the `primary` and `secondary` rows on purpose:
 * HeroUI's chip CSS has compound rules for `primary` against each status color
 * but none against `default`, and `secondary` is an empty rule, so an uncolored
 * chip renders identically under both.
 */
function Matrix() {
  return (
    <div className="flex flex-col gap-3">
      {VARIANTS.map((variant) => (
        <div key={variant} className="flex items-center gap-3">
          <span className="w-20 shrink-0 text-caption">{variant}</span>
          <TrendChip fraction={0.033} polarity="up-is-good" variant={variant} />
          <TrendChip
            fraction={-0.021}
            polarity="up-is-good"
            variant={variant}
          />
          <TrendChip fraction={0} polarity="up-is-good" variant={variant} />
        </div>
      ))}
    </div>
  )
}

export const Variants: Story = { render: () => <Matrix /> }

export const VariantsDark: Story = {
  render: () => <Matrix />,
  globals: { theme: "dark" },
}

/**
 * Three sizes, each scaling the arrow with the text so the glyph keeps reading as
 * part of the line. `sm` is the default: a trend sits beside a headline number,
 * and `md`/`lg` are for the rare case where it is the headline.
 */
export const Sizes: Story = {
  render: () => (
    <div className="flex items-center gap-3">
      <TrendChip fraction={0.033} size="sm" />
      <TrendChip fraction={0.033} size="md" />
      <TrendChip fraction={0.033} size="lg" />
    </div>
  ),
}

/**
 * Polarity is the metric's own axis, not the number's. Spend and error rate
 * improve by falling, so `down-is-good` turns a fall green and a rise red while
 * the arrow keeps telling the truth about which way it went. `neutral` declines
 * to judge, and is the default: most numbers on this dashboard are cost and
 * error rate, so an unconsidered call site should decline rather than paint
 * rising spend green.
 */
export const Polarity: Story = {
  render: () => (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-3">
        <span className="w-28 shrink-0 text-caption">up-is-good</span>
        <TrendChip fraction={0.033} polarity="up-is-good" />
        <TrendChip fraction={-0.021} polarity="up-is-good" />
      </div>
      <div className="flex items-center gap-3">
        <span className="w-28 shrink-0 text-caption">down-is-good</span>
        <TrendChip fraction={0.033} polarity="down-is-good" />
        <TrendChip fraction={-0.021} polarity="down-is-good" />
      </div>
      <div className="flex items-center gap-3">
        <span className="w-28 shrink-0 text-caption">neutral</span>
        <TrendChip fraction={0.033} polarity="neutral" />
        <TrendChip fraction={-0.021} polarity="neutral" />
      </div>
    </div>
  ),
}

/**
 * `caption` carries the comparison window inside the chip, so it is read out with
 * the number rather than hidden in a tooltip. `text` replaces the percentage
 * when the change reads better as an absolute; `fraction` still decides the arrow
 * and the color, which is why the sign is passed even when it is not shown.
 */
export const CaptionAndAbsolute: Story = {
  render: () => (
    <div className="flex flex-col items-start gap-3">
      <TrendChip fraction={-0.059} caption="vs last month" />
      <TrendChip fraction={0.033} caption="MoM" />
      <TrendChip fraction={0.184} text="+$1,234" />
    </div>
  ),
}

/**
 * A change too small to print is flat: no arrow, no status color, and a bare zero
 * rather than a signed one. `fraction={null}` (no comparable previous period,
 * which is what `deltaFraction` returns for an unbounded range) renders nothing
 * at all, so a call site needs no guard of its own.
 */
export const FlatAndAbsent: Story = {
  render: () => (
    <div className="flex items-center gap-3">
      <TrendChip fraction={0} />
      <TrendChip fraction={-0.0004} />
      <TrendChip fraction={null} />
    </div>
  ),
}
