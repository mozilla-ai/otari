import type { Meta, StoryObj } from "@storybook/react-vite"

import { Kbd } from "./Kbd"

/**
 * A keyboard key, in a shortcut hint.
 *
 * `<kbd>` rather than a styled span, which is the whole reason it is a
 * component: the element carries the meaning, and the keys are separate
 * elements so a screen reader reads the chord rather than the "+" between them.
 */
const meta = {
  title: "Design system/Indicators/Kbd",
  component: Kbd,
  args: { keys: ["Cmd", "K"] },
  parameters: { layout: "padded" },
} satisfies Meta<typeof Kbd>

export default meta

type Story = StoryObj<typeof meta>

export const Default: Story = {}

/**
 * The chord is an array in press order rather than a parsed string, so a
 * component never has to guess whether "+" is a separator or a key.
 */
export const Chords: Story = {
  render: () => (
    <div className="flex flex-col items-start gap-2">
      <Kbd keys={["Esc"]} />
      <Kbd keys={["Cmd", "K"]} />
      <Kbd keys={["Ctrl", "Shift", "P"]} />
    </div>
  ),
}

/** In the line it actually ships in, beside the thing it triggers. */
export const InContext: Story = {
  render: () => (
    <p className="flex items-center gap-2 text-caption">
      Clear the filter with <Kbd keys={["Esc"]} />
    </p>
  ),
}
