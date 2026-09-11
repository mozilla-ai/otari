import type { Meta, StoryObj } from "@storybook/react-vite"
import { FiInfo, FiTrash2 } from "react-icons/fi"

import { IconButton } from "../actions/IconButton"
import { Tooltip } from "./Tooltip"

/**
 * A short label revealed by hovering or focusing the thing it describes.
 *
 * **A tooltip is never the only channel.** It does not exist on a phone, so
 * anything only it says is unsaid for a touch operator. That makes it right for
 * a *repetition* (spelling out an icon-only control that already has an
 * `aria-label`, giving an exact timestamp beside a relative one) and wrong for
 * anything an operator needs in order to act.
 *
 * It takes its trigger as `children` rather than as a prop, so the trigger keeps
 * its own type: an `IconButton` inside one is still an `IconButton`, with its
 * required label and 44px box intact.
 *
 * Two forms, and the pair of stories below is the whole difference. A node is
 * wrapped in HeroUI's own trigger, a `div` the library reports as a button so a
 * keyboard can reach something that is not a control. A **native** control is
 * passed as a function and spreads the props it is handed, which is what keeps
 * a real button from ending up inside something else the library also calls a
 * button. A HeroUI `Button` cannot take those props (they are the DOM's, its
 * are react-aria's) and so keeps the wrapper.
 */
const meta = {
  title: "Design system/Overlays/Tooltip",
  component: Tooltip,
  args: { content: "Delete this key", children: null },
  parameters: { layout: "centered" },
} satisfies Meta<typeof Tooltip>

export default meta

type Story = StoryObj<typeof meta>

/**
 * The correct use: the tooltip repeats what the button's `aria-label` already
 * says, so a pointer user and a screen reader get the same sentence and a touch
 * user loses nothing.
 */
export const Default: Story = {
  render: () => (
    <Tooltip content="Delete this key">
      <IconButton label="Delete this key" variant="danger">
        <FiTrash2 aria-hidden className="size-4" />
      </IconButton>
    </Tooltip>
  ),
}

/**
 * The function form, for a native control: one element is both the button and
 * the trigger. Tab to it and the tooltip opens; inspect it and there is a single
 * `button`, where the wrapper form nests one inside a `div` that also reports as
 * a button. `RowAction`'s glyph is the shipping example.
 */
export const AControlTakesTheTriggersProps: Story = {
  render: () => (
    <Tooltip content="Delete">
      {(props) => (
        <button
          {...props}
          type="button"
          aria-label="Delete"
          className="text-caption flex size-8 items-center justify-center hover:text-foreground"
        >
          <FiTrash2 aria-hidden className="h-3.5 w-3.5" />
        </button>
      )}
    </Tooltip>
  ),
}

/** Four placements. Focus the trigger with Tab to open one from the keyboard. */
export const Placements: Story = {
  render: () => (
    <div className="grid grid-cols-2 gap-8 p-12">
      {(["top", "bottom", "left", "right"] as const).map((placement) => (
        <Tooltip key={placement} content={placement} placement={placement}>
          <IconButton label={`Explain ${placement}`}>
            <FiInfo aria-hidden className="size-4" />
          </IconButton>
        </Tooltip>
      ))}
    </div>
  ),
}

/**
 * The other honest use: an exact value beside a rounded one. The relative time
 * is on screen for everyone; the tooltip only adds precision.
 */
export const ExactValue: Story = {
  render: () => (
    <Tooltip content="2026-09-08 14:02:11 UTC">
      <span className="font-mono text-mono-caption">6m ago</span>
    </Tooltip>
  ),
}
