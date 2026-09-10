import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"

import { SecretField } from "./SecretField"

/**
 * A value that must not be readable over a shoulder: a provider key, a
 * password, a webhook secret.
 *
 * Masked, revealable, copyable, and with autofill off. That last one is the
 * part a hand-rolled `type="password"` gets wrong: a browser offering the
 * operator's saved password for a field asking for an OpenAI key is how the
 * wrong credential gets stored in a gateway.
 */
const meta = {
  title: "Design system/Forms/SecretField",
  component: SecretField,
  args: { label: "API key", value: "", onChange: () => {} },
  parameters: { layout: "padded" },
} satisfies Meta<typeof SecretField>

export default meta

type Story = StoryObj<typeof meta>

/** Empty, which is what a first-run provider form shows. */
export const Default: Story = {
  render: () => {
    const [value, setValue] = useState("")
    return (
      <SecretField
        label="OpenAI API key"
        value={value}
        onChange={setValue}
        placeholder="sk-…"
      />
    )
  },
}

/** With a value, so the reveal and copy controls are both reachable. */
export const WithValue: Story = {
  render: () => {
    const [value, setValue] = useState("sk-proj-4f8a2c9e1b7d3a6f5e0c8b2d")
    return (
      <SecretField
        label="OpenAI API key"
        value={value}
        onChange={setValue}
        description="Stored encrypted. It is never shown again after saving."
        reserveMessage
      />
    )
  },
}

/**
 * `isInvalid` is what makes `errorMessage` render, on this field as on every
 * other one here: the message is not shown by its presence alone, so a form
 * holding a stale message cannot flash it while the value is being fixed.
 *
 * The message replaces the description rather than adding a row, and
 * `reserveMessage` holds that line open so the form does not move when it
 * appears.
 */
export const Invalid: Story = {
  render: () => {
    const [value, setValue] = useState("hunter2")
    return (
      <SecretField
        label="OpenAI API key"
        value={value}
        onChange={setValue}
        description="Stored encrypted. It is never shown again after saving."
        isInvalid={!value.startsWith("sk-")}
        errorMessage="An OpenAI key starts with sk-."
        reserveMessage
      />
    )
  },
}

/**
 * `isRequired` for a secret a guardrail profile declares it cannot run without.
 * The marker is the label's, so a form of mixed optional and required fields
 * says which is which before anything is typed rather than on submit.
 */
export const Required: Story = {
  args: {
    label: "Patronus API key",
    value: "",
    isRequired: true,
    description: "This profile refuses to run without it.",
  },
}

/**
 * `isDisabled` for a secret the operator may see the shape of but not set:
 * a profile whose credential the deployment supplies, where the form still
 * names the field so its absence is not read as an oversight.
 */
export const Disabled: Story = {
  args: {
    label: "Alinia API key",
    value: "",
    isDisabled: true,
    description: "Supplied by the deployment; not editable here.",
  },
}
