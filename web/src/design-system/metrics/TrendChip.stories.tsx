import type { Meta, StoryObj } from "@storybook/react-vite"

import { TrendChip } from "./TrendChip"

const meta = {
  title: "Design system/Metrics/TrendChip",
  component: TrendChip,
  args: { fraction: 0.033 },
  parameters: { layout: "padded" },
} satisfies Meta<typeof TrendChip>

export default meta

type Story = StoryObj<typeof meta>

const POLARITIES = ["up-is-good", "down-is-good", "neutral"] as const

/**
 * Every polarity against every direction, which is the matrix the design canvas
 * draws. Nothing here names a color: each fill and ink resolves through the
 * tokens, so the **Theme** toolbar switches the whole grid between the light and
 * dark artboards.
 *
 * Polarity is the metric's own axis, not the number's. Spend and error rate
 * improve by falling, so `down-is-good` turns a fall green and a rise red while
 * the arrow keeps telling the truth about which way it went. `neutral` declines
 * to judge, and is the default: most numbers on this dashboard are cost and
 * error rate, so an unconsidered call site should decline rather than paint
 * rising spend green.
 */
function Matrix() {
  return (
    <div className="flex flex-col gap-3">
      {POLARITIES.map((polarity) => (
        <div key={polarity} className="flex items-center gap-3">
          <span className="w-28 shrink-0 text-caption">{polarity}</span>
          <TrendChip fraction={0.033} polarity={polarity} />
          <TrendChip fraction={-0.021} polarity={polarity} />
          <TrendChip fraction={0} polarity={polarity} />
        </div>
      ))}
    </div>
  )
}

export const Default: Story = {}

export const Polarities: Story = { render: () => <Matrix /> }

/**
 * The same grid on the dark artboard. Worth its own entry rather than leaving it
 * to the toolbar: the status inks are separate values per theme, so this is the
 * only place the two sets are reviewable without switching back and forth.
 */
export const MatrixDark: Story = {
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
