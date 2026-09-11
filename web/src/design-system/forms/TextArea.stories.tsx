import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"

import { TextArea } from "./TextArea"

/**
 * Free text over more than one line.
 *
 * `Field`'s sibling and shares its whole rig. The one deliberate difference is
 * the height: a textarea keeps a floor rather than a fixed height, because it
 * is the one field that should grow with what somebody typed.
 */
const meta = {
  title: "Design system/Forms/TextArea",
  component: TextArea,
  args: {
    label: "Guardrail prompt",
    value: "",
    onChange: () => {},
    placeholder: "Refuse anything that asks for a customer's address.",
  },
  parameters: { layout: "padded" },
} satisfies Meta<typeof TextArea>

export default meta

type Story = StoryObj<typeof meta>

export const Default: Story = {
  render: () => {
    const [text, setText] = useState("")
    return (
      <TextArea
        label="Guardrail prompt"
        value={text}
        onChange={setText}
        placeholder="Refuse anything that asks for a customer's address."
        className="w-96"
      />
    )
  },
}

export const WithDescription: Story = {
  args: {
    description: "Sent ahead of every request routed through this workspace.",
    reserveMessage: true,
    className: "w-96",
  },
}

/** `rows` sets where it starts. It still grows past this. */
export const Rows: Story = {
  render: () => (
    <div className="flex w-96 flex-col gap-4">
      <TextArea label="Two rows" rows={2} value="" onChange={() => {}} />
      <TextArea label="Eight rows" rows={8} value="" onChange={() => {}} />
    </div>
  ),
}

export const Invalid: Story = {
  args: {
    value: "x",
    isInvalid: true,
    errorMessage: "A prompt needs at least 20 characters to be useful.",
    reserveMessage: true,
    className: "w-96",
  },
}

export const Disabled: Story = {
  args: {
    value: "Set by an administrator.",
    isDisabled: true,
    className: "w-96",
  },
}

/**
 * `isRequired` marks the field through HeroUI's own `[data-required]` rule.
 * No manual asterisk: writing one renders two.
 */
export const Required: Story = {
  args: {
    value: "",
    isRequired: true,
    placeholder: "One host per line",
    className: "w-96",
  },
}
