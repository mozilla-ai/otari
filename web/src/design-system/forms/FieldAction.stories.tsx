import type { Meta, StoryObj } from "@storybook/react-vite"

import { Button } from "../actions/Button"
import { Field } from "./Field"
import { FieldAction } from "./FieldAction"

/**
 * A child of a control row with no caption of its own, aligned to the input
 * line.
 *
 * The problem it solves is only visible as a comparison, which is what the
 * first story is. A control row is laid out `items-end`, and that bottom-aligns
 * each child's whole box. A field's box includes the caption line it reserves,
 * so a bare button beside it lands level with the *caption* rather than with
 * the input it acts on. Wrapping the button gives it the same reserve, which
 * puts the two input lines on one row.
 *
 * It reserves rather than nudging: no `self-*`, no padding, nothing asking
 * `align-items` for what it cannot do. The height is the caption role's own
 * line height, so retuning the caption carries the action with it.
 */
const meta = {
  title: "Design system/Forms/FieldAction",
  component: FieldAction,
  args: { children: null },
  parameters: { layout: "padded" },
} satisfies Meta<typeof FieldAction>

export default meta

type Story = StoryObj<typeof meta>

/** Without, then with. Only the second has its two input lines level. */
export const WhatItFixes: Story = {
  render: () => (
    <div className="flex flex-col gap-8">
      <div className="flex flex-col gap-1">
        <span className="text-overline">
          bare button, level with the caption
        </span>
        <div className="flex items-end gap-3">
          <Field
            label="Backend URL"
            value="https://sandbox.internal"
            onChange={() => {}}
            description="Reachable from the gateway."
            reserveMessage
          />
          <Button>Test</Button>
        </div>
      </div>
      <div className="flex flex-col gap-1">
        <span className="text-overline">wrapped, level with the input</span>
        <div className="flex items-end gap-3">
          <Field
            label="Backend URL"
            value="https://sandbox.internal"
            onChange={() => {}}
            description="Reachable from the gateway."
            reserveMessage
          />
          <FieldAction>
            <Button>Test</Button>
          </FieldAction>
        </div>
      </div>
    </div>
  ),
}

/**
 * The two limits, both real and both documented on the component.
 *
 * Every field in the row must reserve exactly one caption line: a message long
 * enough to wrap makes that field taller and drops it back out of line, and no
 * reserve here can answer that. And these rows are `flex-wrap`, where
 * `items-end` aligns each flex *line* to its own cross-end, so the alignment is
 * correct per line rather than per row.
 */
export const WhereItStopsWorking: Story = {
  render: () => (
    <div className="flex w-[28rem] flex-wrap items-end gap-3">
      <Field
        label="Backend URL"
        value="https://sandbox.internal"
        onChange={() => {}}
        description="A hint long enough to wrap onto a second line makes this field taller than the reserve, which drops the action back out of line."
        reserveMessage
      />
      <FieldAction>
        <Button>Test</Button>
      </FieldAction>
    </div>
  ),
}
