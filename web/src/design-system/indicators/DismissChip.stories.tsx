import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"

import { DismissChip } from "./DismissChip"

/**
 * An applied filter, with the control that removes it.
 *
 * Its remove button is drawn 24px and takes a 44px hit area on a touch screen,
 * through a `before:` bleed. Wrapped rows must be 44px apart there
 * (`pointer-coarse:gap-y-5` on the row), or two bleeds overlap and a press near
 * the seam dismisses the neighboring filter. See motion-and-access.md.
 */
const meta = {
  title: "Design system/Indicators/DismissChip",
  component: DismissChip,
  args: { label: "Model", value: "gpt-4o-mini", onDismiss: () => {} },
  parameters: { layout: "padded" },
} satisfies Meta<typeof DismissChip>

export default meta

type Story = StoryObj<typeof meta>

export const Default: Story = {}

/** Without a `label`, for a value that names its own dimension. */
export const ValueOnly: Story = {
  args: { label: undefined, value: "Last 7 days" },
}

/** A row of them, which is how they always appear. Press one to remove it. */
export const Row: Story = {
  render: () => {
    const [filters, setFilters] = useState([
      { label: "Model", value: "gpt-4o-mini" },
      { label: "Workspace", value: "checkout-service" },
      { label: "Status", value: "failed" },
      { label: undefined, value: "Last 7 days" },
    ])
    return (
      <div className="flex w-96 flex-wrap items-center gap-2">
        {filters.map((filter) => (
          <DismissChip
            key={`${filter.label}:${filter.value}`}
            label={filter.label}
            value={filter.value}
            onDismiss={() =>
              setFilters((current) => current.filter((f) => f !== filter))
            }
          />
        ))}
        {filters.length === 0 ? (
          <span className="text-caption">No filters applied.</span>
        ) : null}
      </div>
    )
  },
}

/**
 * `dismissLabel` names the target for assistive tech where the label and value
 * together would not. Without it the button falls back to those two.
 */
export const CustomDismissLabel: Story = {
  args: {
    label: "Workspace",
    value: "checkout-service",
    dismissLabel: "Stop filtering by the checkout-service workspace",
  },
}
