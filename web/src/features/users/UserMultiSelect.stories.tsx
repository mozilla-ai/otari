import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"

import { user } from "@/tests/fixtures"

import { UserMultiSelect } from "./UserMultiSelect"

/**
 * Pick several users, for a budget or a policy that applies to a set of them.
 * Like `UserComboBox` it takes `users` as a prop rather than fetching.
 */
const USERS = [
  user({ user_id: "ops@example.com" }),
  user({ user_id: "dev@example.com" }),
  user({ user_id: "data@example.com" }),
  user({ user_id: "ci-bot" }),
  user({ user_id: "release-bot" }),
]

const meta = {
  title: "Dashboard/Users/UserMultiSelect",
  component: UserMultiSelect,
  args: {
    value: [],
    onChange: () => {},
    users: USERS,
    label: "Applies to",
  },
} satisfies Meta<typeof UserMultiSelect>

export default meta

type Story = StoryObj<typeof meta>

export const Empty: Story = {
  render: (args) => {
    const [value, setValue] = useState<string[]>([])
    return (
      <div className="w-[24rem]">
        <UserMultiSelect {...args} value={value} onChange={setValue} />
      </div>
    )
  },
}

export const WithSelection: Story = {
  render: (args) => {
    const [value, setValue] = useState<string[]>(["ops@example.com", "ci-bot"])
    return (
      <div className="flex w-[24rem] flex-col gap-3">
        <UserMultiSelect {...args} value={value} onChange={setValue} />
        <p className="text-caption">Selected: {value.join(", ") || "none"}</p>
      </div>
    )
  },
}

export const WithDescription: Story = {
  args: {
    value: ["dev@example.com"],
    description:
      "Leave empty to apply the budget to every user in the workspace.",
  },
  render: (args) => (
    <div className="w-[24rem]">
      <UserMultiSelect {...args} />
    </div>
  ),
}

export const NoUsers: Story = {
  args: { users: [] },
  render: (args) => (
    <div className="w-[24rem]">
      <UserMultiSelect {...args} />
    </div>
  ),
}
