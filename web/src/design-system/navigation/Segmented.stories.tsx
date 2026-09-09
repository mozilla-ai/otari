import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"

import { Segmented } from "./Segmented"

/**
 * A closed choice of four or fewer, in a track.
 *
 * The line against `FilterSelect` is the option count: at four or fewer, a
 * track shows every alternative without a click and costs less than a popover.
 * Past four it stops fitting a toolbar and `FilterSelect` takes over.
 *
 * The line against `TabRow`: tabs switch which *view* is on screen, a segmented
 * control changes a *parameter* of the view that is already there. A page can
 * have both, and the usage page does.
 */
const meta = {
  title: "Design system/Navigation/Segmented",
  component: Segmented,
  args: {
    label: "Window",
    value: "7d",
    onChange: () => {},
    options: [
      { value: "24h", label: "24h" },
      { value: "7d", label: "7d" },
      { value: "30d", label: "30d" },
    ],
  },
  parameters: { layout: "padded" },
} satisfies Meta<typeof Segmented>

export default meta

type Story = StoryObj<typeof meta>

export const Default: Story = {
  render: () => {
    const [value, setValue] = useState("7d")
    return (
      <Segmented
        label="Window"
        value={value}
        onChange={setValue}
        options={[
          { value: "24h", label: "24h" },
          { value: "7d", label: "7d" },
          { value: "30d", label: "30d" },
        ]}
      />
    )
  },
}

/** Two options, which is the shape a boolean-ish parameter takes. */
export const TwoOptions: Story = {
  args: {
    label: "Granularity",
    value: "daily",
    options: [
      { value: "hourly", label: "Hourly" },
      { value: "daily", label: "Daily" },
    ],
  },
}

/** Four, which is the documented ceiling. A fifth belongs in a `FilterSelect`. */
export const FourOptions: Story = {
  args: {
    label: "Window",
    value: "30d",
    options: [
      { value: "24h", label: "24h" },
      { value: "7d", label: "7d" },
      { value: "30d", label: "30d" },
      { value: "all", label: "All time" },
    ],
  },
}

/** The selected track on the dark artboard, where its fill is its own value. */
export const Dark: Story = { globals: { theme: "dark" } }
