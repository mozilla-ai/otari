import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"

import { FilterSelect } from "./FilterSelect"

// Required props on the meta, so a story that supplies its own `render` still
// satisfies the component's contract without restating them.
const meta = {
  title: "Design system/Navigation/FilterSelect",
  component: FilterSelect,
  args: { value: "", onChange: () => {}, options: [] },
} satisfies Meta<typeof FilterSelect>

export default meta

type Story = StoryObj<typeof meta>

const STATUSES = [
  { value: "", label: "Any status" },
  { value: "ok", label: "Succeeded" },
  { value: "error", label: "Failed" },
  { value: "refused", label: "Refused" },
]

/**
 * HeroUI's `Select`, so the option list opens as a popover anchored under the
 * trigger rather than a native menu drawn over it. The list here is short; the
 * long ones go through `FilterMultiComboBox` instead.
 *
 * With `label`, the control gets a visible label.
 */
export const WithLabel: Story = {
  render: () => {
    const [value, setValue] = useState("")
    return (
      <FilterSelect
        label="Status"
        value={value}
        onChange={setValue}
        options={STATUSES}
      />
    )
  },
}

/**
 * `ariaLabel` alone gives a compact control with no visible label, for a filter
 * bar where the options name themselves.
 */
export const LabelledOnlyForScreenReaders: Story = {
  render: () => {
    const [value, setValue] = useState("24h")
    return (
      <FilterSelect
        ariaLabel="Time window"
        value={value}
        onChange={setValue}
        options={[
          { value: "1h", label: "Last hour" },
          { value: "24h", label: "Last 24 hours" },
          { value: "7d", label: "Last 7 days" },
          { value: "30d", label: "Last 30 days" },
        ]}
      />
    )
  },
}

export const Disabled: Story = {
  render: () => (
    <FilterSelect
      label="Workspace"
      value=""
      onChange={() => {}}
      disabled
      options={[{ value: "", label: "Only one workspace" }]}
    />
  ),
}

/** A filter bar's worth of them, side by side. */
export const InFilterBar: Story = {
  render: () => {
    const [status, setStatus] = useState("")
    const [window, setWindow] = useState("24h")
    return (
      <div className="flex flex-wrap items-end gap-3">
        <FilterSelect
          label="Status"
          value={status}
          onChange={setStatus}
          options={STATUSES}
        />
        <FilterSelect
          label="Window"
          value={window}
          onChange={setWindow}
          options={[
            { value: "1h", label: "Last hour" },
            { value: "24h", label: "Last 24 hours" },
            { value: "7d", label: "Last 7 days" },
          ]}
        />
      </div>
    )
  },
}

/**
 * `id` when something outside the control has to point at it: a `<label
 * htmlFor>` the filter bar owns rather than the control's own label, or a
 * `aria-describedby` on a hint beside it.
 *
 * Passing `id` does not name the control on its own. The label below is
 * associated through `htmlFor`, which is what makes the two one control to a
 * screen reader; `ariaLabel` is the alternative when there is no visible text
 * to associate.
 */
export const WithExternalLabel: Story = {
  render: () => {
    const [value, setValue] = useState("error")
    return (
      <div className="flex items-center gap-2">
        <label htmlFor="activity-status" className="text-caption">
          Status
        </label>
        <FilterSelect
          id="activity-status"
          value={value}
          onChange={setValue}
          options={STATUSES}
        />
      </div>
    )
  },
}

/**
 * `fullWidth` fills whatever the trigger sits in, which is what a settings row
 * wants: its control lane is a fixed-width slot, and a trigger sized to its own
 * longest option gives the column a different left edge on every row. A filter
 * bar leaves it off, because there the trigger's width naming the filter it
 * carries is the point.
 *
 * A value longer than the trigger is ellipsized rather than allowed to widen
 * it; the popover still shows each option in full.
 */
export const FullWidth: Story = {
  render: () => {
    const [value, setValue] = useState("blocked")
    return (
      <div className="w-[17.5rem]">
        <FilterSelect
          fullWidth
          ariaLabel="Web search for this workspace"
          value={value}
          onChange={setValue}
          options={[
            { value: "default", label: "Deployment default" },
            { value: "allowed", label: "Allowed" },
            { value: "blocked", label: "Blocked (tool and /api/v1/search)" },
          ]}
        />
      </div>
    )
  },
}
