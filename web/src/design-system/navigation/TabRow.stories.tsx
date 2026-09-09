import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"

import { Tab, TabRow } from "./TabRow"

/**
 * A row of views over the same data, and the lane they sit in.
 *
 * Documented as a pair because neither is useful alone: `TabRow` owns the lane
 * and its rule, `Tab` owns one item. They ship from one module for the same
 * reason `RowAction` ships with `RowActionRow`.
 *
 * Controlled, and the active tab is the caller's state. On a real page that
 * state usually lives in the URL through `useUrlState`, so a reload keeps the
 * view; a `Tab` that tracked its own selection would put it out of reach.
 */
const meta = {
  title: "Design system/Navigation/TabRow",
  component: TabRow,
  args: { children: null },
  parameters: { layout: "padded" },
} satisfies Meta<typeof TabRow>

export default meta

type Story = StoryObj<typeof meta>

export const Default: Story = {
  render: () => {
    const [active, setActive] = useState("overview")
    const tabs = [
      ["overview", "Overview"],
      ["by-model", "By model"],
      ["by-workspace", "By workspace"],
      ["by-key", "By key"],
    ]
    return (
      <TabRow>
        {tabs.map(([id, label]) => (
          <Tab key={id} isActive={active === id} onPress={() => setActive(id)}>
            {label}
          </Tab>
        ))}
      </TabRow>
    )
  },
}

/** Two tabs, which is the smallest row worth drawing. Below that, use neither. */
export const TwoTabs: Story = {
  render: () => {
    const [active, setActive] = useState("keys")
    return (
      <TabRow>
        <Tab isActive={active === "keys"} onPress={() => setActive("keys")}>
          Keys
        </Tab>
        <Tab
          isActive={active === "passkeys"}
          onPress={() => setActive("passkeys")}
        >
          Passkeys
        </Tab>
      </TabRow>
    )
  },
}

/** The active item on the dark artboard, where the accent is its own value. */
export const Dark: Story = {
  render: () => (
    <TabRow>
      <Tab isActive onPress={() => {}}>
        Overview
      </Tab>
      <Tab isActive={false} onPress={() => {}}>
        By model
      </Tab>
      <Tab isActive={false} onPress={() => {}}>
        By workspace
      </Tab>
    </TabRow>
  ),
  globals: { theme: "dark" },
}
