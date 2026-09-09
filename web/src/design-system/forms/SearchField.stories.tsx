import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"

import { SearchField } from "./SearchField"

/**
 * A search box: a magnifier, a field, and a clear button once there is
 * something to clear.
 *
 * Filtering is always the caller's. This reports what was typed and nothing
 * else, so a page can debounce it, push it into the URL, or hand it to a query.
 */
const meta = {
  title: "Design system/Forms/SearchField",
  component: SearchField,
  args: { label: "Search settings", value: "", onChange: () => {} },
  parameters: { layout: "padded" },
} satisfies Meta<typeof SearchField>

export default meta

type Story = StoryObj<typeof meta>

/**
 * Type something to reveal the clear button, then press Escape: the field
 * clears itself, which is behavior `type="search"` gives and a text input with
 * a magnifier beside it does not.
 */
export const Default: Story = {
  render: () => {
    const [query, setQuery] = useState("")
    return (
      <SearchField
        label="Search settings"
        value={query}
        onChange={setQuery}
        className="w-72"
      />
    )
  },
}

/** With a value, so the clear button is visible without typing. */
export const WithValue: Story = {
  args: { value: "model_cache", className: "w-72" },
}

/**
 * `label` is the accessible name and is required. The placeholder is not one:
 * it disappears the moment somebody types, taking the field's only name with it.
 */
export const CustomPlaceholder: Story = {
  args: {
    label: "Search models",
    placeholder: "Filter 412 models",
    className: "w-72",
  },
}

export const Disabled: Story = { args: { isDisabled: true, className: "w-72" } }
