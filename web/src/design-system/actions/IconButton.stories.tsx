import type { Meta, StoryObj } from "@storybook/react-vite"
import { FiDownload, FiMoreHorizontal, FiTrash2, FiX } from "react-icons/fi"

import { IconButton } from "./IconButton"

/**
 * A button whose whole label is a glyph.
 *
 * `label` is required rather than an optional `aria-label`, which is the reason
 * this exists as its own component: every story below has a name because there
 * is no way to write one that does not.
 */
const meta = {
  title: "Design system/Actions/IconButton",
  component: IconButton,
  args: {
    label: "Dismiss",
    children: <FiX aria-hidden className="size-4" />,
  },
  parameters: { layout: "padded" },
} satisfies Meta<typeof IconButton>

export default meta

type Story = StoryObj<typeof meta>

export const Default: Story = {}

/**
 * The three variants, all keeping the 44x44 box. An icon-only ghost never takes
 * the edge, and that rule is keyed on what the control *is* rather than where it
 * sits, so no container can reach it.
 */
export const Variants: Story = {
  render: () => (
    <div className="flex items-center gap-3">
      <IconButton label="Export as CSV" variant="primary">
        <FiDownload aria-hidden className="size-4" />
      </IconButton>
      <IconButton label="More actions">
        <FiMoreHorizontal aria-hidden className="size-4" />
      </IconButton>
      <IconButton label="Delete this key" variant="danger">
        <FiTrash2 aria-hidden className="size-4" />
      </IconButton>
    </div>
  ),
}

/**
 * The target is on the box, not in the padding, so it stays at the touch floor
 * at every size. `sm` shrinks the glyph's own button height and keeps the 44px
 * minimum, which is why the three below are the same size on screen.
 */
export const Sizes: Story = {
  render: () => (
    <div className="flex items-center gap-3">
      <IconButton label="Dismiss" size="sm">
        <FiX aria-hidden className="h-3.5 w-3.5" />
      </IconButton>
      <IconButton label="Dismiss" size="md">
        <FiX aria-hidden className="size-4" />
      </IconButton>
      <IconButton label="Dismiss" size="lg">
        <FiX aria-hidden className="size-4" />
      </IconButton>
    </div>
  ),
}

export const Disabled: Story = { args: { isDisabled: true } }
