import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"

import { user } from "@/tests/fixtures"

import { UserComboBox } from "./UserComboBox"

/**
 * The owner picker: choose an existing user, or type an id that does not exist yet.
 *
 * It takes `users` as a prop rather than fetching them, which is what lets the
 * same control serve the keys and routing pages without each of them growing its
 * own copy of the list. The one thing it does read for itself is the organization
 * roster, which is how a member's row reads as their name with the id they are
 * billed under beneath it; these stories run against no roster, so every row here
 * is a plain id.
 *
 * Fixtures come from `src/tests/fixtures.ts`, so a field the gateway adds to
 * `User` arrives in these stories at the same time as in the tests.
 */
const USERS = [
  user({ user_id: "ops@example.com" }),
  user({ user_id: "dev@example.com" }),
  user({ user_id: "data@example.com" }),
  user({ user_id: "ci-bot" }),
]

const meta = {
  title: "Dashboard/Users/UserComboBox",
  component: UserComboBox,
  args: { value: "", onChange: () => {}, users: USERS },
} satisfies Meta<typeof UserComboBox>

export default meta

type Story = StoryObj<typeof meta>

export const Empty: Story = {
  render: (args) => {
    const [value, setValue] = useState("")
    return (
      <div className="w-[24rem]">
        <UserComboBox {...args} value={value} onChange={setValue} />
      </div>
    )
  },
}

export const WithSelection: Story = {
  render: (args) => {
    const [value, setValue] = useState("dev@example.com")
    return (
      <div className="w-[24rem]">
        <UserComboBox {...args} value={value} onChange={setValue} />
      </div>
    )
  },
}

export const WithDescription: Story = {
  args: {
    value: "ops@example.com",
    description: "The key is attributed to this user in usage and activity.",
  },
  render: (args) => (
    <div className="w-[24rem]">
      <UserComboBox {...args} />
    </div>
  ),
}

/**
 * A value that is not in the list. `unknownHint` is what explains it, rather than
 * the control silently showing an id nobody recognizes.
 */
export const UnknownValue: Story = {
  args: {
    value: "retired@example.com",
    unknownHint: "This user no longer exists in the directory.",
  },
  render: (args) => (
    <div className="w-[24rem]">
      <UserComboBox {...args} />
    </div>
  ),
}

/** No users yet, which is a fresh gateway. */
export const NoUsers: Story = {
  args: { users: [], placeholder: "Type a user id" },
  render: (args) => (
    <div className="w-[24rem]">
      <UserComboBox {...args} />
    </div>
  ),
}
