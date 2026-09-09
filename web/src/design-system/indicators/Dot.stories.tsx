import type { Meta, StoryObj } from "@storybook/react-vite"

import { Dot } from "./Dot"

/**
 * A 6px square. The page's one status mark, in every place it appears.
 *
 * It takes its fill as a required `className`, which looks unusual and is
 * deliberate: the dot has no opinion about what it means, and a `tone` prop
 * would have to enumerate the status hues *and* the eight chart slots, which are
 * two vocabularies that must not be mixed. `src/styles/dotRamp.test.ts` is what
 * holds that line, failing on a status dot given a surface-ramp value.
 *
 * **A dot is never the only channel.** Every story below has a word beside it,
 * because a red-green deficiency removes the only channel a bare square has.
 */
const meta = {
  title: "Design system/Indicators/Dot",
  component: Dot,
  args: { className: "bg-success" },
  parameters: { layout: "padded" },
} satisfies Meta<typeof Dot>

export default meta

type Story = StoryObj<typeof meta>

export const Default: Story = {
  render: () => (
    <span className="flex items-center gap-2 text-caption">
      <Dot className="bg-success" />
      Healthy
    </span>
  ),
}

/** The status hues, each with the word that carries the meaning. */
export const StatusHues: Story = {
  render: () => (
    <div className="flex flex-col gap-2">
      {[
        ["bg-success", "Active"],
        ["bg-warning", "Near limit"],
        ["bg-danger", "Over budget"],
        ["bg-info", "Read-only"],
        ["bg-text-subtle", "Not configured"],
      ].map(([fill, word]) => (
        <span key={word} className="flex items-center gap-2 text-caption">
          <Dot className={fill} />
          {word}
        </span>
      ))}
    </div>
  ),
}

/**
 * The categorical chart slots, which are a separate vocabulary. Assigned in
 * order and never cycled: the color-deficiency margins are computed on adjacent
 * pairs in exactly this order, so wrapping around invalidates them.
 *
 * Spelled out one literal at a time rather than built from an index, and that
 * is not verbosity. Tailwind emits only the utilities it can *see* in the
 * source, and it does not evaluate a template literal, so `bg-chart-cat-${n}`
 * compiles, ships, and paints eight colorless squares. Every dynamic class in
 * this tree has the same failure mode and it is silent.
 */
export const ChartSlots: Story = {
  render: () => (
    <div className="flex flex-col gap-2">
      {(
        [
          ["bg-chart-cat-1", "cat-1"],
          ["bg-chart-cat-2", "cat-2"],
          ["bg-chart-cat-3", "cat-3"],
          ["bg-chart-cat-4", "cat-4"],
          ["bg-chart-cat-5", "cat-5"],
          ["bg-chart-cat-6", "cat-6"],
          ["bg-chart-cat-7", "cat-7"],
          ["bg-chart-cat-8", "cat-8"],
          [
            "bg-chart-cat-other",
            "cat-other, the neutral slot a ninth group folds into",
          ],
        ] as const
      ).map(([fill, name]) => (
        <span key={name} className="flex items-center gap-2 text-caption">
          <Dot className={fill} />
          {name}
        </span>
      ))}
    </div>
  ),
}
