import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"

import { MultiSelect } from "./MultiSelect"

const PEOPLE = [
  { id: "operator", label: "Operator" },
  { id: "alice@example.com", label: "alice@example.com" },
  { id: "parity-heavy@example.com", label: "parity-heavy@example.com" },
  { id: "parity-light@example.com", label: "parity-light@example.com" },
  { id: "pat@example.com", label: "pat@example.com (Pat Okafor)" },
  { id: "priya@example.com", label: "priya@example.com (Priya Raghavan)" },
  { id: "sam@example.com", label: "sam@example.com (Sam Iyer)" },
  { id: "wei@example.com", label: "wei@example.com (Wei Zhang)" },
]

/**
 * Pick several without the field moving.
 *
 * Controlled, like every other control here, so each story owns the selection it
 * shows. The interesting states are what the selection does to the layout, so
 * they differ by how much is picked rather than by a prop.
 */
const meta = {
  title: "Design system/Forms/MultiSelect",
  component: MultiSelect,
  parameters: { layout: "padded" },
  args: {
    label: "Assign to people (optional)",
    description:
      "Everyone selected is held to this budget, each with their own allowance rather than a shared pool.",
    options: PEOPLE,
    value: [],
    onChange: () => {},
    searchPlaceholder: "Search people…",
    countNoun: "people assigned",
  },
} satisfies Meta<typeof MultiSelect>

export default meta

type Story = StoryObj<typeof meta>

function Live({
  initial,
  ...args
}: React.ComponentProps<typeof MultiSelect> & { initial?: string[] }) {
  const [value, setValue] = useState<string[]>(initial ?? [])
  return <MultiSelect {...args} value={value} onChange={setValue} />
}

/** Nothing picked. The field reads as a search box, because that is all it is. */
export const Empty: Story = {
  render: (args) => <Live {...args} />,
}

/**
 * Six picked. The field has not moved: the chips are below it, so the block
 * grew downward and the control the operator is pointing at stayed put.
 */
export const WithSelection: Story = {
  render: (args) => (
    <Live
      {...args}
      initial={[
        "operator",
        "alice@example.com",
        "parity-heavy@example.com",
        "parity-light@example.com",
        "pat@example.com",
        "priya@example.com",
      ]}
    />
  ),
}

/**
 * The list open. A picked person stays listed with a check rather than
 * disappearing, so the list answers "who is in" and not only "who is left", and
 * the row order does not change on a pick.
 *
 * Open it and type to see the footer count follow the query. Stories cannot
 * carry an open popover on their own: it opens on focus, which a static render
 * has not had.
 */
export const Open: Story = {
  render: (args) => (
    <Live {...args} initial={["parity-heavy@example.com", "operator"]} />
  ),
}

/** A refusal, on the same rung a `Field`'s sits on. */
export const Invalid: Story = {
  render: (args) => (
    <Live
      {...args}
      isInvalid
      errorMessage="Pick at least one person, or leave the budget unassigned."
      reserveMessage
    />
  ),
}

/**
 * Nothing matches the query, which is a different sentence from having nothing
 * to offer: one is about what was typed, the other about the deployment.
 */
export const NoMatches: Story = {
  args: {
    noMatchesMessage: "Nobody matches what you typed.",
  },
  render: (args) => <Live {...args} />,
}

/** Nothing to offer, which is a different sentence from nothing matching. */
export const NoOptions: Story = {
  args: {
    options: [],
    emptyMessage:
      "Nobody to assign yet. Add people under Members & roles, or issue a key, and they can be assigned here.",
  },
  render: (args) => <Live {...args} />,
}
