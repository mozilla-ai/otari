import type { Meta, StoryObj } from "@storybook/react-vite"

import { RowAction, RowActionRow } from "./RowAction"

/**
 * An action in a table row, and the trailing lane those sit in.
 *
 * Not a `Button`, and that is the rule rather than a preference: a row of
 * buttons in every table row turns a dense table into a grid of boxes, so a row
 * action is caption-sized text. `RowActionRow` is the lane; reach for it rather
 * than `deprecated/RowActions`, which is a near-duplicate with a tighter gap
 * still on one call site.
 *
 * The danger prop paints the danger ink, and `src/styles/dotRamp.test.ts`
 * enforces something about it worth knowing: a row action never paints danger
 * *at rest*. A table whose every row carries a red "Delete" reads as a table of
 * problems, so the hue is spent on the armed step of a confirm instead, which
 * is why `ConfirmRowAction` supplies it and a call site here does not.
 *
 * That gate reads raw lines rather than parsed JSX, so naming the prop in prose
 * here would land this file on its offender list. Hence "the danger prop".
 */
const meta = {
  title: "Design system/Actions/RowAction",
  component: RowAction,
  args: { onPress: () => {}, children: "Edit" },
  parameters: { layout: "padded" },
} satisfies Meta<typeof RowAction>

export default meta

type Story = StoryObj<typeof meta>

export const Default: Story = {}

/** The lane, which is how they always appear. */
export const InALane: Story = {
  render: () => (
    <RowActionRow>
      <RowAction onPress={() => {}}>Edit</RowAction>
      <RowAction onPress={() => {}}>Rotate</RowAction>
      <RowAction onPress={() => {}}>Revoke</RowAction>
    </RowActionRow>
  ),
}

/**
 * `isDisabled` with an `ariaLabel` carrying the reason. That pairing is the
 * point: a disabled control takes no focus, so a tooltip reaches a pointer and
 * nothing else, and the reason has to be in the accessible name.
 */
export const DisabledWithReason: Story = {
  render: () => (
    <RowActionRow>
      <RowAction onPress={() => {}}>Edit</RowAction>
      <RowAction
        onPress={() => {}}
        isDisabled
        ariaLabel="Revoke this key. Unavailable: an organization owner manages it."
      >
        Revoke
      </RowAction>
    </RowActionRow>
  ),
}

/** In the row it ships in, trailing the data. */
export const InContext: Story = {
  render: () => (
    <div className="flex w-[32rem] flex-col">
      {["checkout-service", "batch-ingest", "staging-probe"].map((name) => (
        <div
          key={name}
          className="flex items-center gap-4 border-b border-border py-2"
        >
          <span className="min-w-0 flex-1 font-mono text-mono-caption">
            {name}
          </span>
          <RowActionRow>
            <RowAction onPress={() => {}} ariaLabel={`Edit ${name}`}>
              Edit
            </RowAction>
            <RowAction onPress={() => {}} ariaLabel={`Revoke ${name}`}>
              Revoke
            </RowAction>
          </RowActionRow>
        </div>
      ))}
    </div>
  ),
}
