import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"
import { RefreshButton } from "../actions/RefreshButton"
import { FilterChips } from "./FilterChips"
import { FilterMultiComboBox } from "./FilterMultiComboBox"
import { FilterSelect } from "./FilterSelect"

const meta = {
  title: "Design system/Navigation/FilterChips",
  component: FilterChips,
  args: { chips: [], children: null },
  parameters: { layout: "padded" },
} satisfies Meta<typeof FilterChips>

export default meta

type Story = StoryObj<typeof meta>

/**
 * Nothing applied: just the "Add filter" disclosure holding the pickers. The
 * pickers live inside the disclosure; the applied set lives outside it as chips,
 * so what is filtering the page stays visible whether or not the picker is open.
 */
export const NoFilters: Story = {
  args: {
    children: (
      <FilterSelect
        label="Status"
        value=""
        onChange={() => {}}
        options={[
          { value: "", label: "Any status" },
          { value: "error", label: "Failed" },
        ]}
      />
    ),
  },
}

export const WithChips: Story = {
  args: {
    chips: [
      { key: "status", label: "Status", value: "Failed", onClear: () => {} },
      {
        key: "model",
        label: "Model",
        value: "openai:gpt-4o-mini",
        onClear: () => {},
      },
    ],
    onClearAll: () => {},
    children: null,
  },
}

/** `start` and `end` slot a page's own controls into the same row. */
export const WithSlots: Story = {
  args: {
    chips: [
      {
        key: "window",
        label: "Window",
        value: "Last 24 hours",
        onClear: () => {},
      },
    ],
    start: <span className="text-overline">Activity</span>,
    end: <RefreshButton onRefresh={() => {}} updatedAt={Date.now() - 45_000} />,
    children: null,
  },
}

/** Many chips wrap rather than pushing the row wider. */
export const ManyChips: Story = {
  args: {
    onClearAll: () => {},
    chips: [
      { key: "status", label: "Status", value: "Failed", onClear: () => {} },
      {
        key: "model",
        label: "Model",
        value: "openai:gpt-4o-mini",
        onClear: () => {},
      },
      {
        key: "model2",
        label: "Model",
        value: "anthropic:claude-haiku-4-5",
        onClear: () => {},
      },
      {
        key: "user",
        label: "User",
        value: "ops@example.com",
        onClear: () => {},
      },
      { key: "key", label: "Key", value: "ci-pipeline", onClear: () => {} },
      {
        key: "window",
        label: "Window",
        value: "Last 7 days",
        onClear: () => {},
      },
    ],
    children: null,
  },
}

/**
 * The whole loop wired up: pick in the disclosure, the chip appears, clearing the
 * chip removes it. This is how a page composes the two, `FilterMultiComboBox`
 * deliberately has no chip row of its own.
 */
export const Interactive: Story = {
  render: () => {
    const [models, setModels] = useState<string[]>(["openai:gpt-4o-mini"])
    const [status, setStatus] = useState("")
    const options = [
      "openai:gpt-4o",
      "openai:gpt-4o-mini",
      "anthropic:claude-haiku-4-5",
    ].map((value) => ({ value, label: value }))

    const chips = [
      ...models.map((value) => ({
        key: `model:${value}`,
        label: "Model",
        value,
        onClear: () =>
          setModels((current) => current.filter((m) => m !== value)),
      })),
      ...(status
        ? [
            {
              key: "status",
              label: "Status",
              value: status === "error" ? "Failed" : "Succeeded",
              onClear: () => setStatus(""),
            },
          ]
        : []),
    ]

    return (
      <FilterChips
        chips={chips}
        onClearAll={
          chips.length
            ? () => {
                setModels([])
                setStatus("")
              }
            : undefined
        }
      >
        <FilterMultiComboBox
          label="Models"
          values={models}
          onChange={setModels}
          options={options}
          placeholder="All models"
        />
        <FilterSelect
          label="Status"
          value={status}
          onChange={setStatus}
          options={[
            { value: "", label: "Any status" },
            { value: "ok", label: "Succeeded" },
            { value: "error", label: "Failed" },
          ]}
        />
      </FilterChips>
    )
  },
}
