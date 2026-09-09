import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"

import { Field } from "./Field"
import { ControlField, FieldMessages } from "./FieldMessages"

/**
 * The line under a field, and the space it takes whether or not it speaks.
 *
 * A field's description and its error are **one role at one size, separated by
 * color**, so they occupy one line and share one reserve. An error therefore
 * *replaces* the description rather than adding a row, which is what keeps a
 * form from growing as it is validated.
 *
 * `ControlField` is the same label-and-description rig for a control that is not
 * a HeroUI field: a segmented group, a multi-select, a scope picker. It exists
 * because that construct was hand-written each time as a `text-sm` label over a
 * `text-xs` description, which is how the size drifted below the caption in
 * eight files at once.
 */
const meta = {
  title: "Design system/Forms/FieldMessages",
  component: FieldMessages,
  args: { children: null },
  parameters: { layout: "padded" },
} satisfies Meta<typeof FieldMessages>

export default meta

type Story = StoryObj<typeof meta>

/**
 * The reserve, which is the whole point. Type into the second field and then
 * clear it: the row below never moves, because the line was already there.
 * The first field has no reserve, so the form jumps.
 */
export const TheReserve: Story = {
  render: () => {
    const [without, setWithout] = useState("")
    const [with_, setWith] = useState("")
    return (
      <div className="flex w-80 flex-col gap-8">
        <div className="flex flex-col gap-1">
          <span className="text-overline">reserveMessage off</span>
          <Field
            label="Key name"
            value={without}
            onChange={setWithout}
            isInvalid={without.length > 0 && without.length < 3}
            errorMessage="At least 3 characters."
          />
          <div className="border-t border-border pt-2 text-caption">
            This line moves when the message appears.
          </div>
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-overline">reserveMessage on</span>
          <Field
            label="Key name"
            value={with_}
            onChange={setWith}
            isInvalid={with_.length > 0 && with_.length < 3}
            errorMessage="At least 3 characters."
            reserveMessage
          />
          <div className="border-t border-border pt-2 text-caption">
            This line does not.
          </div>
        </div>
      </div>
    )
  },
}

/**
 * `ControlField` around something that is not a field. With no description
 * there is no reserve: an empty line under a control that will never say
 * anything is space held for nothing.
 */
export const ControlFieldRig: Story = {
  render: () => (
    <div className="flex w-80 flex-col gap-6">
      <ControlField
        label="Model scope"
        description="Which models a caller using this key may reach."
      >
        <div className="border border-border p-3 text-caption">
          a control that is not a HeroUI field
        </div>
      </ControlField>
      <ControlField label="No description, so no reserve">
        <div className="border border-border p-3 text-caption">another one</div>
      </ControlField>
    </div>
  ),
}
