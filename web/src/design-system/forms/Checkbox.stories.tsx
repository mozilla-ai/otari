import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"

import { Checkbox, CheckboxVisual } from "./Checkbox"

// Required props on the meta, so a story that supplies its own `render` still
// satisfies the component's contract without restating them.
const meta = {
  title: "Design system/Forms/Checkbox",
  component: Checkbox,
  args: {
    isSelected: false,
    onChange: () => {},
    children: "Refresh pricing on startup",
  },
} satisfies Meta<typeof Checkbox>

export default meta

type Story = StoryObj<typeof meta>

/**
 * Controlled, which is the only way it comes: the box owns no state of its own
 * beyond the optimistic press flash.
 */
export const Default: Story = {
  render: () => {
    const [on, setOn] = useState(false)
    return (
      <Checkbox isSelected={on} onChange={setOn}>
        Refresh pricing on startup
      </Checkbox>
    )
  },
}

export const Checked: Story = {
  render: () => {
    const [on, setOn] = useState(true)
    return (
      <Checkbox isSelected={on} onChange={setOn}>
        Refresh pricing on startup
      </Checkbox>
    )
  },
}

export const Disabled: Story = {
  render: () => (
    <div className="flex flex-col gap-3">
      <Checkbox isSelected={false} onChange={() => {}} isDisabled>
        Unavailable in this deployment
      </Checkbox>
      <Checkbox isSelected onChange={() => {}} isDisabled>
        Enforced by the platform
      </Checkbox>
    </div>
  ),
}

/**
 * The box glyph on its own, in all four states.
 *
 * `CheckboxVisual` is split out of `Checkbox` so `DataTable`'s selection column
 * can share it, one visual serves both, which is what keeps a standalone
 * checkbox and a table's selection box from drifting apart. Indeterminate is
 * reachable only here and in a table's header box.
 */
export const Visuals: Story = {
  render: () => (
    <div className="flex items-center gap-6">
      {[
        {
          label: "off",
          isSelected: false,
          isIndeterminate: false,
          isDisabled: false,
        },
        {
          label: "on",
          isSelected: true,
          isIndeterminate: false,
          isDisabled: false,
        },
        {
          label: "mixed",
          isSelected: false,
          isIndeterminate: true,
          isDisabled: false,
        },
        {
          label: "disabled",
          isSelected: false,
          isIndeterminate: false,
          isDisabled: true,
        },
      ].map((state) => (
        <span key={state.label} className="flex flex-col items-center gap-2">
          <CheckboxVisual
            isSelected={state.isSelected}
            isIndeterminate={state.isIndeterminate}
            isDisabled={state.isDisabled}
          />
          <span className="text-caption">{state.label}</span>
        </span>
      ))}
    </div>
  ),
}

/**
 * `ariaLabel` when the visible label alone does not say which control it is.
 *
 * These three read "Allowed" on screen, which is all a column heading needs;
 * spoken on its own it names nothing. The accessible name keeps the visible
 * text inside it, so speech input still reaches the control by what is written
 * on it.
 */
export const NamedForScreenReaders: Story = {
  render: () => {
    const [allowed, setAllowed] = useState<string[]>(["staging"])
    return (
      <div className="flex flex-col gap-3">
        {["production", "staging", "sandbox"].map((workspace) => (
          <span key={workspace} className="flex items-center gap-3">
            <span className="w-24 text-caption">{workspace}</span>
            <Checkbox
              isSelected={allowed.includes(workspace)}
              onChange={(on) =>
                setAllowed((current) =>
                  on
                    ? [...current, workspace]
                    : current.filter((name) => name !== workspace),
                )
              }
              ariaLabel={`Allowed in ${workspace}`}
            >
              Allowed
            </Checkbox>
          </span>
        ))}
      </div>
    )
  },
}

/**
 * `hasTouchTarget` for a list a thumb operates.
 *
 * The box stays 16px and nothing beside it moves; the label claims 44px so the
 * press has somewhere to land. `CopyButton`'s pseudo-element bleed is the usual
 * answer to the same problem and is wrong here: a 16px box needs 14px each way
 * to reach 44, which in a list of hairline-divided rows would overlap the row
 * above and below and send a press near the seam to the wrong one.
 */
export const TouchTarget: Story = {
  render: () => {
    const [picked, setPicked] = useState<string[]>([])
    return (
      <ul className="flex w-64 flex-col">
        {["prod-gateway", "ci-bot", "analytics-ro"].map((key) => (
          <li
            key={key}
            className="flex items-center gap-2 border-b border-border-subtle"
          >
            <Checkbox
              hasTouchTarget
              isSelected={picked.includes(key)}
              onChange={(on) =>
                setPicked((current) =>
                  on ? [...current, key] : current.filter((n) => n !== key),
                )
              }
              ariaLabel={`Select ${key}`}
            >
              <span className="sr-only">Select {key}</span>
            </Checkbox>
            <span className="text-body">{key}</span>
          </li>
        ))}
      </ul>
    )
  },
}
