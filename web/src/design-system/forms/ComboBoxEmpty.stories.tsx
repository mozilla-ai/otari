import type { Meta, StoryObj } from "@storybook/react-vite"

import { ComboBoxEmpty } from "./ComboBoxEmpty"

/**
 * What a combo box's popover says with nothing in it.
 *
 * Shown here on its own rather than inside a field, because a story cannot hold
 * a popover open: `ComboBoxField` owns whether its menu is open, and these two
 * sentences are the whole of what this renders. The frame below stands in for
 * the popover's own edge.
 */
const meta = {
  title: "Design system/Forms/ComboBoxEmpty",
  component: ComboBoxEmpty,
  parameters: { layout: "padded" },
  decorators: [
    (Story) => (
      <div className="w-72 border border-control-border">
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof ComboBoxEmpty>

export default meta

type Story = StoryObj<typeof meta>

/** The query matched none of the options the source does have. Typing less fixes it. */
export const NoMatches: Story = {}

/** The same state with the caller's own wording, for a list whose match rules are not "contains". */
export const NoMatchesCustom: Story = {
  args: { noMatchesMessage: "No policy matches that name." },
}

/**
 * The source itself is empty, which on the routing page is what a gateway with no
 * provider credential looks like. The caller supplies the sentence, because only
 * it knows what would fill the list.
 */
export const NothingYet: Story = {
  args: {
    isSourceEmpty: true,
    emptyMessage:
      "No models discovered yet. Add a provider credential and the models it serves appear here.",
  },
}

/** Its default, for a caller with nothing more useful to say. */
export const NothingYetDefault: Story = { args: { isSourceEmpty: true } }
