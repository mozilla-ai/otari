import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"

import { ComboBoxField, type ComboBoxOption } from "./ComboBoxField"

const MODELS: ComboBoxOption[] = [
  { value: "openai:gpt-4o", label: "openai:gpt-4o" },
  { value: "openai:gpt-4o-mini", label: "openai:gpt-4o-mini" },
  {
    value: "anthropic:claude-sonnet-4-5",
    label: "anthropic:claude-sonnet-4-5",
  },
  {
    value: "mistral:mistral-large",
    label: "mistral:mistral-large",
    isDisabled: true,
  },
]

/**
 * One of a set, searchable, in a form the operator submits.
 *
 * The combo box counterpart to `Select`, sharing its prop vocabulary. Open a
 * story's menu to see the list; the two empty sentences have their own stories
 * on `ComboBoxEmpty`, since a story cannot hold this one's popover open.
 */
const meta = {
  title: "Design system/Forms/ComboBoxField",
  component: ComboBoxField,
  args: {
    label: "Serves",
    value: "",
    onChange: () => {},
    options: MODELS,
    className: "w-72",
  },
  parameters: { layout: "padded" },
} satisfies Meta<typeof ComboBoxField>

export default meta

type Story = StoryObj<typeof meta>

/**
 * `value` is the picked option's `value` and the input shows its `label`, the
 * same division `Select` makes. The field filters nothing: `onQueryChange`
 * publishes what is in the input and the caller decides what matches.
 */
export const Default: Story = {
  render: (args) => {
    const [value, setValue] = useState("")
    const [query, setQuery] = useState("")
    const match = query.trim().toLowerCase()
    return (
      <ComboBoxField
        {...args}
        value={value}
        onChange={setValue}
        onQueryChange={setQuery}
        options={MODELS.filter((option) =>
          option.label.toLowerCase().includes(match),
        )}
      />
    )
  },
}

/**
 * A description explains the field, and an option's `isDisabled` shows a choice
 * that exists but cannot be taken. `reserveMessage` holds the caption line open
 * so a message appearing does not move the form.
 */
export const WithDescription: Story = {
  args: {
    value: "openai:gpt-4o",
    description: "Requests naming this model are routed to it directly.",
    reserveMessage: true,
  },
}

/**
 * `hint` is the option's secondary line: text that identifies the label rather
 * than repeating it, such as the id a person is billed under. The value here is
 * an id, and the box shows the name it belongs to.
 */
export const OptionHints: Story = {
  args: {
    label: "Owner",
    value: "018f0000-0000-4000-8000-000000000001",
    placeholder: "Pick a user, or type a new id…",
    options: [
      {
        value: "018f0000-0000-4000-8000-000000000001",
        label: "Ada Lovelace",
        hint: "018f0000-0000-4000-8000-000000000001",
      },
      {
        value: "018f0000-0000-4000-8000-000000000002",
        label: "Grace Hopper",
        hint: "018f0000-0000-4000-8000-000000000002",
      },
      { value: "ci-bot", label: "ci-bot" },
    ],
  },
}

/**
 * An error replaces the description rather than adding a row, which is why they
 * share one line.
 */
export const Invalid: Story = {
  args: {
    description: "Requests naming this model are routed to it directly.",
    isRequired: true,
    isInvalid: true,
    errorMessage: "Name a model before saving.",
    reserveMessage: true,
  },
}

export const Disabled: Story = {
  args: { value: "openai:gpt-4o", isDisabled: true },
}

/**
 * Free text, for a list that is a shortcut rather than a whitelist: model
 * discovery only sees what the configured credentials expose, so a model it
 * cannot list must stay typeable.
 *
 * `menuTrigger="input"` goes with `autoFocus`, and only with it: react-aria
 * marks everything outside an open popover aria-hidden, so a list that opens on
 * arrival would hide the rest of the form from a screen reader before a single
 * keystroke.
 */
export const CustomValueAutoFocused: Story = {
  args: {
    value: "openrouter:vendor/model-7",
    allowsCustomValue: true,
    autoFocus: true,
    menuTrigger: "input",
    description: "Anything typed here stands, listed or not.",
  },
}

/**
 * `shouldSelectOnFocus` for a field whose input shows the current selection: typing
 * then replaces it rather than appending to a name that filters to nothing.
 *
 * The empty sentences are passed here too. Which one applies is `isSourceEmpty`,
 * which the caller answers because only it knows whether its list is empty
 * because of the query or because there is nothing to offer.
 */
export const PickFromList: Story = {
  args: {
    value: "anthropic:claude-sonnet-4-5",
    shouldSelectOnFocus: true,
    menuTrigger: "focus",
    isSourceEmpty: false,
    emptyMessage: "No models discovered yet. Add a provider credential.",
    noMatchesMessage: "No model matches. Type a selector to use it anyway.",
  },
}
