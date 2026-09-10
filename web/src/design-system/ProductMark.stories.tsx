import type { Meta, StoryObj } from "@storybook/react-vite"

import { ProductMark } from "./ProductMark"

/**
 * The product mark.
 *
 * At the top level of the design system rather than in a topic directory, and
 * that is deliberate: it belongs to no topic, and a directory holding one file
 * is worse than a file. It is also the one place inline SVG is right, because it
 * is the mark rather than an icon; icons come from `react-icons/fi`, where a
 * hand-rolled glyph would drift invisibly from the library one it imitates.
 *
 * Sized by the caller through `className`, since the rail, the sign-in screen
 * and the top bar each want a different size and the mark has no opinion.
 */
const meta = {
  title: "Design system/ProductMark",
  component: ProductMark,
  parameters: { layout: "padded" },
} satisfies Meta<typeof ProductMark>

export default meta

type Story = StoryObj<typeof meta>

export const Default: Story = {}

/** The sizes it ships at, smallest first. */
export const Sizes: Story = {
  render: () => (
    <div className="flex items-end gap-6">
      {(["h-4", "h-6", "h-8", "h-12"] as const).map((height) => (
        <div key={height} className="flex flex-col items-center gap-2">
          <ProductMark className={height} />
          <span className="text-caption">{height}</span>
        </div>
      ))}
    </div>
  ),
}

/** On the dark artboard: the mark reads the accent token, so it retheme with it. */
export const Dark: Story = {
  render: () => <ProductMark className="h-8" />,
  globals: { theme: "dark" },
}
