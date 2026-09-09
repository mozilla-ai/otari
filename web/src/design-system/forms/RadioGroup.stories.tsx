import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"

import { RadioGroup } from "./RadioGroup"

const RETENTION = [
  {
    value: "30d",
    label: "30 days",
    description: "The default. Enough to answer a billing question.",
  },
  {
    value: "1y",
    label: "1 year",
    description: "For a deployment that reports usage quarterly.",
  },
  {
    value: "forever",
    label: "Keep everything",
    description: "Rows are never pruned. The table grows without bound.",
  },
]

/**
 * One of a short set, with every option on screen at once.
 *
 * The line against `Select`: a radio group spends vertical space to make the
 * alternatives readable, so it is for a choice whose options need explaining or
 * where seeing all of them at once is the point. Past about five, or where the
 * labels are self-evident, `Select` is the smaller control.
 */
const meta = {
  title: "Design system/Forms/RadioGroup",
  component: RadioGroup,
  args: {
    label: "Usage retention",
    value: "30d",
    onChange: () => {},
    options: RETENTION,
  },
  parameters: { layout: "padded" },
} satisfies Meta<typeof RadioGroup>

export default meta

type Story = StoryObj<typeof meta>

/** The case this control is for: three options that each need a sentence. */
export const Default: Story = {
  render: () => {
    const [value, setValue] = useState("30d")
    return (
      <RadioGroup
        label="Usage retention"
        value={value}
        onChange={setValue}
        options={RETENTION}
        description="Applies to every workspace in this deployment."
        className="w-96"
      />
    )
  },
}

/** Without per-option descriptions, which is where `Select` is usually better. */
export const LabelsOnly: Story = {
  args: {
    label: "Scope",
    value: "workspace",
    options: [
      { value: "workspace", label: "This workspace" },
      { value: "organization", label: "Every workspace" },
    ],
    className: "w-96",
  },
}

/** `horizontal` only for two or three short labels; it wraps badly past that. */
export const Horizontal: Story = {
  args: {
    label: "Scope",
    value: "workspace",
    orientation: "horizontal",
    options: [
      { value: "workspace", label: "This workspace" },
      { value: "organization", label: "Every workspace" },
    ],
    className: "w-96",
  },
}

/** An option that exists but cannot be taken, shown rather than hidden. */
export const OptionDisabled: Story = {
  args: {
    label: "Scope",
    value: "workspace",
    options: [
      { value: "workspace", label: "This workspace" },
      {
        value: "organization",
        label: "Every workspace",
        description: "Needs an organization owner.",
        isDisabled: true,
      },
    ],
    className: "w-96",
  },
}

export const Invalid: Story = {
  args: {
    value: "",
    isInvalid: true,
    errorMessage: "Pick a retention window.",
    className: "w-96",
  },
}
