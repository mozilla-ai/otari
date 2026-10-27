import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"

import { CheckboxGroup, type CheckboxOption } from "./CheckboxGroup"

const WEEKDAYS = [
  { value: "1", label: "Monday" },
  { value: "2", label: "Tuesday" },
  { value: "3", label: "Wednesday" },
  { value: "4", label: "Thursday" },
  { value: "5", label: "Friday" },
  { value: "6", label: "Saturday" },
  { value: "0", label: "Sunday" },
]

const SIGNALS = [
  {
    value: "email",
    label: "Email",
    description: "To the organization's owners and admins.",
  },
  {
    value: "banner",
    label: "In-app banner",
    description: "Shown while the budget is over its threshold.",
  },
  {
    value: "webhook",
    label: "Webhook",
    description: "Not configured for this deployment.",
    isDisabled: true,
  },
]

/** Several of a short set, with every option on screen at once. See design/forms.md. */
const meta = {
  title: "Design system/Forms/CheckboxGroup",
  component: CheckboxGroup,
  args: { label: "Reset on", value: [], onChange: () => {}, options: WEEKDAYS },
  parameters: { layout: "padded" },
} satisfies Meta<typeof CheckboxGroup>

export default meta
type Story = StoryObj<typeof meta>

function Controlled({
  options,
  initial = [],
  ...rest
}: {
  options: readonly CheckboxOption[]
  initial?: string[]
} & Partial<Parameters<typeof CheckboxGroup>[0]>) {
  const [value, setValue] = useState<string[]>(initial)
  return (
    <CheckboxGroup
      label="Reset on"
      value={value}
      onChange={setValue}
      options={options}
      {...rest}
    />
  )
}

/** The case it was built for: a weekly cycle's selected weekdays. */
export const Default: Story = {
  render: () => <Controlled options={WEEKDAYS} initial={["1", "5"]} />,
}

/** Horizontal, for labels short enough to sit in a row. */
export const Horizontal: Story = {
  render: () => (
    <Controlled
      options={WEEKDAYS.map((day) => ({
        ...day,
        label: day.label.slice(0, 3),
      }))}
      initial={["1", "5"]}
      orientation="horizontal"
    />
  ),
}

/** A per-option description, and one option the deployment cannot offer. */
export const WithDescriptions: Story = {
  render: () => (
    <Controlled
      options={SIGNALS}
      initial={["email"]}
      label="Alert by"
      description="Each threshold sends once per period."
    />
  ),
}

/** Invalid, with the message announced on the group rather than beside it. */
export const Invalid: Story = {
  render: () => (
    <Controlled
      options={WEEKDAYS}
      isInvalid
      errorMessage="Pick at least one weekday."
    />
  ),
}

/**
 * Required: at least one option must be picked. react-aria sets
 * `aria-required` on the group; the label carries no visual mark.
 */
export const Required: Story = {
  render: () => (
    <Controlled
      options={WEEKDAYS}
      initial={["1"]}
      isRequired
      description="A weekly cycle resets on each of the days picked here."
    />
  ),
}

/**
 * The label kept for assistive technology where the surrounding control already
 * shows it. The weekday group inside a reset-cycle picker is the case: the
 * picker names the cycle, so a second visible "Reset on" would say it twice.
 */
export const LabelHidden: Story = {
  render: () => (
    <div className="flex flex-col gap-2">
      <span className="text-body">Reset cycle</span>
      <Controlled
        options={WEEKDAYS.map((day) => ({
          ...day,
          label: day.label.slice(0, 3),
        }))}
        initial={["1", "5"]}
        hideLabel
        orientation="horizontal"
      />
    </div>
  ),
}

/** Every option disabled, for a caller who may read the group but not change it. */
export const Disabled: Story = {
  render: () => (
    <Controlled options={WEEKDAYS} initial={["1", "5"]} isDisabled />
  ),
}
