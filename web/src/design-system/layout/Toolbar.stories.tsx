import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"

import { Button } from "../actions/Button"
import { SearchField } from "../forms/SearchField"
import { FilterSelect } from "../navigation/FilterSelect"
import { Toolbar } from "./Toolbar"

/**
 * The row of filters and actions above a table.
 *
 * **It is a place, not a container.** That is the one thing worth understanding
 * here: `Toolbar` puts `.otari-toolbar` on the row, and the controls inside it
 * take the dense 32px height and drop a ghost button's edge *because of where
 * they are*, not because any call site asked. A page says "this row is a
 * toolbar" and every control in it agrees on a size; no call site picks a
 * height, which is what stopped the six pages with a filter row from each
 * picking a different one.
 *
 * The same mechanism raises everything back to 44px below `md`, so the dense
 * height is a desktop size that the phone layout takes off again.
 */
const meta = {
  title: "Design system/Layout/Toolbar",
  component: Toolbar,
  args: { children: null },
  parameters: { layout: "padded" },
} satisfies Meta<typeof Toolbar>

export default meta

type Story = StoryObj<typeof meta>

/** Resize the canvas past 767px to watch every control come up to 44px together. */
export const Default: Story = {
  render: () => {
    const [query, setQuery] = useState("")
    const [status, setStatus] = useState("")
    return (
      <Toolbar>
        <SearchField
          label="Search keys"
          value={query}
          onChange={setQuery}
          className="w-56"
        />
        <FilterSelect
          ariaLabel="Status"
          value={status}
          onChange={setStatus}
          options={[
            { value: "", label: "Any status" },
            { value: "active", label: "Active" },
            { value: "revoked", label: "Revoked" },
          ]}
        />
        <Button className="ml-auto">Export CSV</Button>
      </Toolbar>
    )
  },
}

/**
 * The place's other job: a ghost button inside a toolbar renders edgeless,
 * because a row of edged ghosts reads as a grid of boxes. Compare the two rows.
 */
export const GhostEdgeDropped: Story = {
  render: () => (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-1">
        <span className="text-overline">inside a toolbar</span>
        <Toolbar>
          <Button>Export</Button>
          <Button>Refresh</Button>
          <Button>Columns</Button>
        </Toolbar>
      </div>
      <div className="flex flex-col gap-1">
        <span className="text-overline">the same three, outside one</span>
        <div className="flex items-center gap-2">
          <Button>Export</Button>
          <Button>Refresh</Button>
          <Button>Columns</Button>
        </div>
      </div>
    </div>
  ),
}
