import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"

import { Toggle } from "./Toggle"

/**
 * A boolean that commits when it is flipped.
 *
 * That is the whole line against `Checkbox`, which belongs to a form somebody
 * submits. If there is a Save button, it is a checkbox.
 *
 * Controlled, with no state of its own: the settings page's switches each
 * report to a mutation, and a switch that flipped itself before the request
 * landed would lie about the deployment for as long as the round trip took.
 */
const meta = {
  title: "Design system/Forms/Toggle",
  component: Toggle,
  args: {
    label: "Freeze new dashboard sign-ins",
    isSelected: false,
    onChange: () => {},
  },
  parameters: { layout: "padded" },
} satisfies Meta<typeof Toggle>

export default meta

type Story = StoryObj<typeof meta>

/** The knob's color carries the state; the track is a drawn outline either way. */
export const Default: Story = {
  render: () => {
    const [on, setOn] = useState(false)
    return (
      <Toggle
        label="Freeze new dashboard sign-ins"
        isSelected={on}
        onChange={setOn}
      />
    )
  },
}

/**
 * Both states side by side. On a flat plane the track is an outline rather than
 * a raised trough, so nothing about the track changes: `control-thumb` off,
 * `control-indicator` on, and the travel is what gets the transition.
 */
export const States: Story = {
  render: () => (
    <div className="flex items-center gap-6">
      <Toggle label="Off" isSelected={false} onChange={() => {}} />
      <Toggle label="On" isSelected onChange={() => {}} />
      <Toggle
        label="Disabled, off"
        isSelected={false}
        onChange={() => {}}
        isDisabled
      />
      <Toggle label="Disabled, on" isSelected onChange={() => {}} isDisabled />
    </div>
  ),
}

/**
 * The visible track is 24px and the hit area is 44px, carried by a `before:`
 * pseudo-element rather than by padding so a row's height does not move. Hover
 * the space above and below the track to feel it.
 */
export const TouchTarget: Story = {
  render: () => {
    const [on, setOn] = useState(true)
    return (
      <div className="flex flex-col gap-1">
        <Toggle
          label="Enable model discovery"
          isSelected={on}
          onChange={setOn}
        />
        <p className="text-caption">
          The pressable area extends 10px past the track on both sides.
        </p>
      </div>
    )
  },
}
