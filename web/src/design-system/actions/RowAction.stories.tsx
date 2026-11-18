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
 * An action in a table row, and the lane those sit in.
 *
 * Inside a `RowActionRow`, which is where a table's actions go, each one is an
 * item of the row's three-dot menu: its glyph and its label, written as the
 * menu shows it (`Edit…` when it asks before acting). `isDestructive` puts a
 * hard-to-undo action in the danger ink, after a rule; the menu is closed until
 * somebody opens it, so that red is never on the row at rest.
 *
 * Alone, a row action is still a bare glyph with a 44px target, for the rare
 * control on a row that is not one of its actions.
 *
 * The danger prop paints the danger ink on a standalone action, and
 * `src/styles/dotRamp.test.ts` enforces something about it worth knowing: it is
 * the armed step of a confirm, never the resting state. That gate reads raw
 * lines rather than parsed JSX, so naming the prop in prose here would land this
 * file on its offender list. Hence "the danger prop".
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

/** A row's actions, which is how they always appear: one three-dot menu. */
export const InALane: Story = {
  render: () => (
    <RowActionRow>
      <RowAction icon={FiClock} label="History" onPress={() => {}} />
      <RowAction icon={FiEdit2} label="Edit…" onPress={() => {}} />
      <RowAction icon={FiRefreshCw} label="Regenerate…" onPress={() => {}} />
      <RowAction
        icon={FiTrash2}
        label="Delete…"
        isDestructive
        onPress={() => {}}
      />
    </RowActionRow>
  ),
}

/** A note that is not an action stays on the row, beside the trigger. */
export const WithANote: Story = {
  render: () => (
    <RowActionRow>
      <span className="text-caption">Default</span>
      <RowAction icon={FiEdit2} label="Edit…" onPress={() => {}} />
    </RowActionRow>
  ),
}

/** A toggle carries its state in its label and its glyph. */
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
 * `isDisabled` with an `ariaLabel` carrying the reason. The item shows the
 * reason under its label, and the trigger's own name carries it too: a disabled
 * item takes no focus, so a keyboard would otherwise never reach it.
 */
export const DisabledWithReason: Story = {
  render: () => (
    <RowActionRow>
      <RowAction icon={FiEdit2} label="Edit…" onPress={() => {}} />
      <RowAction
        icon={FiSlash}
        label="Revoke"
        onPress={() => {}}
        isDisabled
        ariaLabel="An organization owner manages this key."
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
              label="Edit…"
              ariaLabel={`Edit ${name}`}
              onPress={() => {}}
            />
            <RowAction
              icon={FiSlash}
              label="Revoke"
              ariaLabel={`Revoke ${name}`}
              isDestructive
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
 * Standalone, because inside a `RowActionRow` the action is a menu item and
 * `ConfirmRowAction` asks in a dialog instead. This story exists because the
 * prop is public, so the catalog owes a picture of what it does.
 */
export const ArmedPaintsDanger: Story = {
  render: () => {
    const [armed, setArmed] = useState(false)
    return (
      <div className="flex flex-col gap-3">
        <RowAction isDanger={armed} onPress={() => setArmed(!armed)}>
          {armed ? "Confirm remove" : "Remove"}
        </RowAction>
        <p className="text-caption">
          Press Remove: the ink is neutral at rest and danger once armed.
        </p>
      </div>
    )
  },
}
