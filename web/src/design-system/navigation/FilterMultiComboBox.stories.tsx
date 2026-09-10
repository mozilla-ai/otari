import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"

import { FilterMultiComboBox } from "./FilterMultiComboBox"

// Required props on the meta, so a story that supplies its own `render` still
// satisfies the component's contract without restating them.
const meta = {
  title: "Design system/Navigation/FilterMultiComboBox",
  component: FilterMultiComboBox,
  args: { label: "Models", values: [], onChange: () => {}, options: [] },
} satisfies Meta<typeof FilterMultiComboBox>

export default meta

type Story = StoryObj<typeof meta>

const MODELS = [
  "openai:gpt-4o",
  "openai:gpt-4o-mini",
  "openai:o3-mini",
  "anthropic:claude-opus-4",
  "anthropic:claude-sonnet-4",
  "anthropic:claude-haiku-4-5",
  "mistral:mistral-large",
  "google:gemini-2.5-pro",
].map((value) => ({ value, label: value }))

/**
 * Type to narrow, press to add.
 *
 * A native `<select>` over thousands of models or users is unusable, and the
 * question a usage view answers is usually a comparison ("these three models"),
 * not a single choice. Picking clears the query and leaves the list open on the
 * remaining options, so a run of selections is one gesture. Picked options drop
 * out of the list.
 *
 * Removal deliberately lives with the page's `FilterChips`, not a second chip row
 * here, so the applied set stays visible whether or not the picker is open.
 */
export const Empty: Story = {
  render: () => {
    const [values, setValues] = useState<string[]>([])
    return (
      <div className="w-[20rem]">
        <FilterMultiComboBox
          label="Models"
          values={values}
          onChange={setValues}
          options={MODELS}
          placeholder="All models"
        />
      </div>
    )
  },
}

/** Once something is picked, the input reports the size of the selection. */
export const WithSelection: Story = {
  render: () => {
    const [values, setValues] = useState<string[]>([
      "openai:gpt-4o-mini",
      "anthropic:claude-haiku-4-5",
    ])
    return (
      <div className="flex w-[20rem] flex-col gap-3">
        <FilterMultiComboBox
          label="Models"
          values={values}
          onChange={setValues}
          options={MODELS}
          placeholder="All models"
        />
        <p className="text-caption">Applied: {values.join(", ") || "none"}</p>
      </div>
    )
  },
}

/**
 * At the ceiling. `maxValues` matches what the analytics endpoints accept for one
 * repeatable filter, so stopping here keeps a 51st pick from failing every query
 * on the page with a 422 the operator cannot read. The remaining options are still
 * offered but inert, so the list reads as full rather than silently swallowing a
 * click.
 */
export const AtLimit: Story = {
  render: () => {
    const [values, setValues] = useState<string[]>([
      "openai:gpt-4o",
      "openai:gpt-4o-mini",
    ])
    return (
      <div className="w-[20rem]">
        <FilterMultiComboBox
          label="Models"
          values={values}
          onChange={setValues}
          options={MODELS}
          maxValues={2}
        />
      </div>
    )
  },
}

/**
 * `allowsCustom` commits typed text on Enter, for a filter whose value space is
 * not enumerable, any model name the log might hold, not just the ones a
 * windowed suggestion list knows. The options stay suggestions.
 */
export const AllowsCustomValues: Story = {
  render: () => {
    const [values, setValues] = useState<string[]>([])
    return (
      <div className="flex w-[20rem] flex-col gap-3">
        <FilterMultiComboBox
          label="Models"
          values={values}
          onChange={setValues}
          options={MODELS}
          placeholder="Any model"
          allowsCustom
        />
        <p className="text-caption">
          Type a name that is not listed and press Enter.
        </p>
      </div>
    )
  },
}

/**
 * `maxVisible` windows the list. With 400 options the picker still narrows as you
 * type rather than rendering them all.
 */
export const LargeOptionList: Story = {
  render: () => {
    const [values, setValues] = useState<string[]>([])
    const many = Array.from({ length: 400 }, (_, index) => ({
      value: `user_${String(index).padStart(4, "0")}`,
      label: `operator-${index}@example.com`,
    }))
    return (
      <div className="w-[20rem]">
        <FilterMultiComboBox
          label="Users"
          values={values}
          onChange={setValues}
          options={many}
          placeholder="All users"
        />
      </div>
    )
  },
}

/**
 * `maxVisible` caps how many options the list renders, defaulting to 50.
 *
 * The cap is why this control can sit on a page listing thousands of models: it
 * narrows as you type rather than rendering the catalog. Set low here so the cap
 * is visible with a list short enough to count. Type to narrow past it.
 */
export const CappedList: Story = {
  render: () => {
    const [values, setValues] = useState<string[]>([])
    const options = Array.from({ length: 40 }, (_, index) => ({
      value: `model-${index}`,
      label: `provider:model-${String(index).padStart(2, "0")}`,
    }))
    return (
      <div className="flex w-[26rem] flex-col gap-2">
        <FilterMultiComboBox
          label="Models"
          values={values}
          onChange={setValues}
          options={options}
          maxVisible={5}
        />
        <p className="text-caption">
          40 options, maxVisible 5. The list shows five until the query narrows
          it.
        </p>
      </div>
    )
  },
}
