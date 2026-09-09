import type { Meta, StoryObj } from "@storybook/react-vite"

import { Divider } from "./Divider"

/**
 * A hairline between two things: the flat plane's only division.
 *
 * A component rather than a `border-t` at a call site because the border family
 * has three rungs with different jobs, and spelling one inline is how a page
 * ends up with whichever rung somebody copied. All three are alpha rather than
 * opaque, so one line composites correctly on all four surface rungs; an opaque
 * border tuned to the canvas is invisible on a table body.
 *
 * A `Section`'s own rules are not this. A band draws its own edges.
 */
const meta = {
  title: "Design system/Layout/Divider",
  component: Divider,
  parameters: { layout: "padded" },
} satisfies Meta<typeof Divider>

export default meta

type Story = StoryObj<typeof meta>

export const Default: Story = {
  render: () => (
    <div className="w-96">
      <Divider />
    </div>
  ),
}

/**
 * The three weights, on the page ground. The difference is deliberately small:
 * 3% inside one group, 6% between bands, 10% for a division that has to read as
 * structural.
 */
export const Weights: Story = {
  render: () => (
    <div className="flex w-96 flex-col gap-6">
      {(["subtle", "default", "strong"] as const).map((weight) => (
        <div key={weight} className="flex flex-col gap-2">
          <span className="text-overline">{weight}</span>
          <Divider weight={weight} />
        </div>
      ))}
    </div>
  ),
}

/** The same three on the dark artboard, where each is a separate alpha. */
export const WeightsDark: Story = {
  render: () => (
    <div className="flex w-96 flex-col gap-6">
      {(["subtle", "default", "strong"] as const).map((weight) => (
        <div key={weight} className="flex flex-col gap-2">
          <span className="text-overline">{weight}</span>
          <Divider weight={weight} />
        </div>
      ))}
    </div>
  ),
  globals: { theme: "dark" },
}

/** Vertical, for a divider between two controls in one row. */
export const Vertical: Story = {
  render: () => (
    <div className="flex items-center gap-3 text-caption">
      <span>12 keys</span>
      <Divider orientation="vertical" className="h-4" />
      <span>3 workspaces</span>
      <Divider orientation="vertical" className="h-4" />
      <span>$41.20 this month</span>
    </div>
  ),
}

/**
 * On each surface rung, which is what the alpha is for: the same line reads on
 * all four, where an opaque value tuned to one would disappear on another.
 */
export const OnEverySurface: Story = {
  render: () => (
    <div className="flex w-96 flex-col">
      {(
        [
          ["bg-background", "background"],
          ["bg-surface", "surface"],
          ["bg-surface-alt", "surface-alt"],
          ["bg-surface-subtle", "surface-subtle"],
        ] as const
      ).map(([fill, name]) => (
        <div key={name} className={`flex flex-col gap-2 p-4 ${fill}`}>
          <span className="text-overline">{name}</span>
          <Divider />
        </div>
      ))}
    </div>
  ),
}
