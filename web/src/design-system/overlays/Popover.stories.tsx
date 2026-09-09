import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"

import { Button } from "../actions/Button"
import { Checkbox } from "../forms/Checkbox"
import { Popover } from "./Popover"

/**
 * A panel anchored to the control that opened it.
 *
 * The line against `Tooltip`: a tooltip labels, a popover holds content an
 * operator interacts with, so this one takes focus and is dismissed
 * deliberately. The line against `Dialog`: a dialog is modal and centered
 * because it wants the whole screen, a popover stays anchored because what it
 * says is about the thing it points at.
 *
 * Uncontrolled by default, which is the opposite of `Dialog` and deliberate: a
 * popover's trigger is inside it, so it can own that state.
 */
const meta = {
  title: "Design system/Overlays/Popover",
  component: Popover,
  args: { trigger: null, children: null },
  parameters: { layout: "centered" },
} satisfies Meta<typeof Popover>

export default meta

type Story = StoryObj<typeof meta>

/** Uncontrolled: the trigger opens it, Escape and an outside click close it. */
export const Default: Story = {
  render: () => (
    <Popover trigger={<Button>Columns</Button>}>
      <div className="flex flex-col gap-2 p-1">
        <span className="text-overline">Visible columns</span>
        <Checkbox isSelected onChange={() => {}}>
          Key name
        </Checkbox>
        <Checkbox isSelected onChange={() => {}}>
          Workspace
        </Checkbox>
        <Checkbox isSelected={false} onChange={() => {}}>
          Created
        </Checkbox>
      </div>
    </Popover>
  ),
}

export const Placements: Story = {
  render: () => (
    <div className="grid grid-cols-2 gap-8 p-12">
      {(["top", "bottom", "left", "right"] as const).map((placement) => (
        <Popover
          key={placement}
          placement={placement}
          trigger={<Button>{placement}</Button>}
        >
          <p className="p-1 text-body">Anchored {placement}.</p>
        </Popover>
      ))}
    </div>
  ),
}

/**
 * Controlled, for the case where something other than the trigger closes it: a
 * route change, or a mutation landing. Here the button inside it does.
 */
export const Controlled: Story = {
  render: () => {
    const [isOpen, setIsOpen] = useState(false)
    return (
      <Popover
        isOpen={isOpen}
        onOpenChange={setIsOpen}
        trigger={<Button>Reassign workspace</Button>}
      >
        <div className="flex flex-col gap-3 p-1">
          <p className="text-body">Move this key to another workspace?</p>
          <Button variant="primary" onPress={() => setIsOpen(false)}>
            Move it
          </Button>
        </div>
      </Popover>
    )
  },
}
