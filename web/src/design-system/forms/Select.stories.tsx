import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"

import { Select } from "./Select"

const STRATEGIES = [
  { value: "fallback", label: "Fallback" },
  { value: "round-robin", label: "Round robin" },
  { value: "least-cost", label: "Least cost" },
  { value: "weighted", label: "Weighted", isDisabled: true },
]

/**
 * One of a short, closed set, in a form the operator submits.
 *
 * The form counterpart to `navigation/FilterSelect`. Two components rather than
 * one with a mode, because a filter's label is a caption beside the control and
 * never speaks a validation message, where this one's label sits above it and
 * owns a description and an error line announced on the control.
 */
const meta = {
  title: "Design system/Forms/Select",
  component: Select,
  args: {
    label: "Routing strategy",
    value: "fallback",
    onChange: () => {},
    options: STRATEGIES,
  },
  parameters: { layout: "padded" },
} satisfies Meta<typeof Select>

export default meta

type Story = StoryObj<typeof meta>

export const Default: Story = {
  render: () => {
    const [value, setValue] = useState("fallback")
    return (
      <Select
        label="Routing strategy"
        value={value}
        onChange={setValue}
        options={STRATEGIES}
        className="w-72"
      />
    )
  },
}

/** A description explains the field; an option's `isDisabled` shows a choice that exists but cannot be taken. */
export const WithDescription: Story = {
  args: {
    description: "Applies to every model in this workspace.",
    reserveMessage: true,
    className: "w-72",
  },
}

/**
 * An error replaces the description rather than adding a row, which is why they
 * share one line. `reserveMessage` holds that line open so the form does not
 * move when the message appears.
 */
export const Invalid: Story = {
  args: {
    value: "",
    description: "Applies to every model in this workspace.",
    isInvalid: true,
    errorMessage: "Pick a strategy before saving.",
    reserveMessage: true,
    className: "w-72",
  },
}

/**
 * A value no option carries, which is what a URL naming a withdrawn choice
 * produces. It is carried as its own option rather than letting react-aria
 * print "Select an item", so the field says what is actually set.
 */
export const UnknownValue: Story = {
  args: { value: "cheapest-first", className: "w-72" },
}

export const Disabled: Story = { args: { isDisabled: true, className: "w-72" } }

/**
 * `isRequired` marks the field through HeroUI's own `[data-required]` rule, so
 * nothing here spells the asterisk out; writing one renders two.
 *
 * `placeholder` is what the trigger says while nothing is selected, and it is
 * an example rather than a restatement of the label. The default reads "Select
 * an option", which is what an unset field with nothing better to say shows.
 */
export const Required: Story = {
  args: {
    value: "",
    isRequired: true,
    placeholder: "Pick how failures are routed",
    className: "w-72",
  },
}

/**
 * `autoFocus` puts the caret in a form's first field, the way `ComboBoxField`'s
 * does. A dialog whose first control is a picker takes it; a page's filter
 * never should, since focus on arrival belongs to the page.
 */
export const AutoFocused: Story = {
  args: {
    value: "",
    autoFocus: true,
    placeholder: "Pick how failures are routed",
    className: "w-72",
  },
}
