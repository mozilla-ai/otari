import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"
import {
  FiCheckCircle,
  FiClock,
  FiEdit2,
  FiRefreshCw,
  FiSlash,
  FiTrash2,
} from "react-icons/fi"

import { RowAction, RowActionRow } from "./RowAction"

/**
 * An action in a table row, and the trailing lane those sit in.
 *
 * Not a `Button`, and that is the rule rather than a preference: a row of
 * buttons in every table row turns a dense table into a grid of boxes, so a row
 * action wears a bare glyph. `RowActionRow` is the lane; reach for it rather
 * than `deprecated/RowActions`, which is a near-duplicate with a tighter gap
 * still on one call site.
 *
 * The glyph form is what a lane should reach for, and the word is not lost to
 * it: `label` is the accessible name and the tooltip both, so it is still read
 * aloud, still matched by speech input, and still spelled out on hover. The
 * text form is left for what no glyph says, which in practice is the armed half
 * of a confirm.
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
  args: { onPress: () => {}, icon: FiEdit2, label: "Edit" },
  parameters: { layout: "padded" },
} satisfies Meta<typeof RowAction>

export default meta

type Story = StoryObj<typeof meta>

export const Default: Story = {}

/** The lane, which is how they always appear. */
export const InALane: Story = {
  render: () => (
    <RowActionRow>
      <RowAction icon={FiClock} label="History" onPress={() => {}} />
      <RowAction icon={FiEdit2} label="Edit" onPress={() => {}} />
      <RowAction icon={FiRefreshCw} label="Regenerate" onPress={() => {}} />
      <RowAction icon={FiTrash2} label="Delete" onPress={() => {}} />
    </RowActionRow>
  ),
}

/**
 * The words the glyphs replaced, for comparison: four of them per row, re-read
 * on every row, against four shapes that are recognized rather than read
 * (otari-ai#2123). The text form is still what a confirm's armed half uses,
 * which is why it is public rather than retired.
 */
export const TheTextFormItReplaced: Story = {
  render: () => (
    <RowActionRow>
      <RowAction onPress={() => {}}>History</RowAction>
      <RowAction onPress={() => {}}>Edit</RowAction>
      <RowAction onPress={() => {}}>Regenerate</RowAction>
      <RowAction onPress={() => {}}>Delete</RowAction>
    </RowActionRow>
  ),
}

/**
 * A pair whose glyph carries the state as well as the act, which is what a
 * label alone cannot do: the lane says at a glance which rows are blocked.
 */
export const AToggledAction: Story = {
  render: () => {
    const [blocked, setBlocked] = useState(false)
    return (
      <RowActionRow>
        <RowAction
          icon={blocked ? FiCheckCircle : FiSlash}
          label={blocked ? "Unblock" : "Block"}
          onPress={() => setBlocked(!blocked)}
        />
      </RowActionRow>
    )
  },
}

/**
 * `isDisabled` with an `ariaLabel` carrying the reason. That pairing is the
 * point: a disabled control takes no focus, so a tooltip reaches a pointer and
 * nothing else, and the reason has to be in the accessible name.
 */
export const DisabledWithReason: Story = {
  render: () => (
    <RowActionRow>
      <RowAction icon={FiEdit2} label="Edit" onPress={() => {}} />
      <RowAction
        icon={FiSlash}
        label="Revoke"
        onPress={() => {}}
        isDisabled
        ariaLabel="Revoke this key. Unavailable: an organization owner manages it."
      />
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
            <RowAction
              icon={FiEdit2}
              label="Edit"
              ariaLabel={`Edit ${name}`}
              onPress={() => {}}
            />
            <RowAction
              icon={FiSlash}
              label="Revoke"
              ariaLabel={`Revoke ${name}`}
              onPress={() => {}}
            />
          </RowActionRow>
        </div>
      ))}
    </div>
  ),
}

/**
 * The danger ink, and the rule about when it may appear.
 *
 * `src/styles/dotRamp.test.ts` rejects a call site that paints it
 * unconditionally: a table whose every row carries a red "Delete" reads as a
 * table of problems, so the hue is spent on the armed step of a confirm rather
 * than at rest. The prop below is bound to an armed flag, which is the only
 * shape that gate allows and also the only one worth showing.
 *
 * `ConfirmRowAction` is what a row should actually reach for. This story exists
 * because the prop is public, so the catalog owes a picture of what it does.
 */
export const ArmedPaintsDanger: Story = {
  render: () => {
    const [armed, setArmed] = useState(false)
    return (
      <div className="flex flex-col gap-3">
        <RowActionRow>
          <RowAction icon={FiEdit2} label="Edit" onPress={() => {}} />
          <RowAction isDanger={armed} onPress={() => setArmed(!armed)}>
            {armed ? "Confirm remove" : "Remove"}
          </RowAction>
        </RowActionRow>
        <p className="text-caption">
          Press Remove: the ink is neutral at rest and danger once armed.
        </p>
      </div>
    )
  },
}
