import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"

import { Field } from "./Field"

const meta = {
  title: "Design system/Forms/Field",
  component: Field,
  args: {
    label: "Key name",
    value: "",
    onChange: () => {},
  },
} satisfies Meta<typeof Field>

export default meta

type Story = StoryObj<typeof meta>

/**
 * The label is a real `<label>` wired to the input by HeroUI's `TextField`, so it
 * is queryable the way an operator reads it. The field caps itself at `max-w-md`
 * rather than stretching to whatever container it lands in.
 */
export const Default: Story = {
  args: { placeholder: "ci-pipeline" },
}

export const WithValue: Story = {
  args: { value: "ci-pipeline" },
}

export const Required: Story = {
  args: { value: "ci-pipeline", isRequired: true },
}

/** `description` is a node, so it can carry an inline code sample or a link. */
export const WithDescription: Story = {
  args: {
    value: "ci-pipeline",
    description: (
      <>
        Shown in usage and activity. Not the key itself, which is generated for
        you.
      </>
    ),
  },
}

/**
 * The only other `type` it takes. A budget window's start and end are the reason
 * it exists.
 */
export const DateTime: Story = {
  args: {
    label: "Resets at",
    type: "datetime-local",
    value: "2026-09-01T00:00",
    description: "Interpreted in the gateway's timezone.",
  },
}

/** Typing into it, and stacked the way a create form uses it. */
export const InForm: Story = {
  render: () => {
    const [name, setName] = useState("")
    const [budget, setBudget] = useState("")
    return (
      <div className="flex w-[28rem] flex-col gap-4">
        <Field
          label="Key name"
          value={name}
          onChange={setName}
          placeholder="ci-pipeline"
          isRequired
        />
        <Field
          label="Monthly cap (USD)"
          value={budget}
          onChange={setBudget}
          placeholder="100"
          description="Leave empty for no cap."
        />
      </div>
    )
  },
}

/**
 * `autoFocus` puts the caret in the field on mount, for the one field a dialog
 * or a first-run form exists to collect. Reload the story to see it take: focus
 * happens once, so switching to this story from another one does not repeat it.
 *
 * One per screen. Two fields both claiming the caret means the second wins and
 * the first looks broken, and a page that steals focus on every render takes it
 * away from whatever the operator was doing.
 */
export const AutoFocused: Story = {
  args: { autoFocus: true, description: "The caret starts here." },
}

/**
 * `isDisabled` for a field the operator may read but not set: a value the
 * deployment fixes, or one that belongs to a plan they are not on. It stays in
 * the tab order's reading path but takes no input, which is why the value is
 * left legible rather than dimmed to the point of being unreadable.
 */
export const Disabled: Story = {
  args: {
    isDisabled: true,
    value: "gateway-managed",
    description: "Set by the deployment, not per workspace.",
  },
}
