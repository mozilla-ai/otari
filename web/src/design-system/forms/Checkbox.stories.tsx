import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"

import { Checkbox, CheckboxVisual } from "./Checkbox"

// Required props on the meta, so a story that supplies its own `render` still
// satisfies the component's contract without restating them.
const meta = {
  title: "Design system/Forms/Checkbox",
  component: Checkbox,
  args: {
    isSelected: false,
    onChange: () => {},
    children: "Refresh pricing on startup",
  },
} satisfies Meta<typeof Checkbox>

export default meta

type Story = StoryObj<typeof meta>

/**
 * Controlled, which is the only way it comes: the box owns no state of its own
 * beyond the optimistic press flash.
 */
export const Default: Story = {
  render: () => {
    const [on, setOn] = useState(false)
    return (
      <Checkbox isSelected={on} onChange={setOn}>
        Refresh pricing on startup
      </Checkbox>
    )
  },
}

export const Checked: Story = {
  render: () => {
    const [on, setOn] = useState(true)
    return (
      <Checkbox isSelected={on} onChange={setOn}>
        Refresh pricing on startup
      </Checkbox>
    )
  },
}

export const Disabled: Story = {
  render: () => (
    <div className="flex flex-col gap-3">
      <Checkbox isSelected={false} onChange={() => {}} isDisabled>
        Unavailable in this deployment
      </Checkbox>
      <Checkbox isSelected onChange={() => {}} isDisabled>
        Enforced by the platform
      </Checkbox>
    </div>
  ),
}

/**
 * The box glyph on its own, in all four states.
 *
 * `CheckboxVisual` is split out of `Checkbox` so `DataTable`'s selection column
 * can share it, one visual serves both, which is what keeps a standalone
 * checkbox and a table's selection box from drifting apart. Indeterminate is
 * reachable only here and in a table's header box.
 */
export const Visuals: Story = {
  render: () => (
    <div className="flex items-center gap-6">
      {[
        {
          label: "off",
          isSelected: false,
          isIndeterminate: false,
          isDisabled: false,
        },
        {
          label: "on",
          isSelected: true,
          isIndeterminate: false,
          isDisabled: false,
        },
        {
          label: "mixed",
          isSelected: false,
          isIndeterminate: true,
          isDisabled: false,
        },
        {
          label: "disabled",
          isSelected: false,
          isIndeterminate: false,
          isDisabled: true,
        },
      ].map((state) => (
        <span key={state.label} className="flex flex-col items-center gap-2">
          <CheckboxVisual
            isSelected={state.isSelected}
            isIndeterminate={state.isIndeterminate}
            isDisabled={state.isDisabled}
          />
          <span className="text-caption">{state.label}</span>
        </span>
      ))}
    </div>
  ),
}
