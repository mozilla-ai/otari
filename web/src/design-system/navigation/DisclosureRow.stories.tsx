import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"

import { Toggle } from "../forms/Toggle"
import { SettingRow } from "../layout/SettingRow"
import { SettingsGroup } from "../layout/SettingsGroup"
import { DisclosureRow } from "./DisclosureRow"

/**
 * A settings row that opens in place, for detail belonging to the row rather
 * than beside it.
 *
 * **The whole row is the button**, which is what keeps the target at the row's
 * own size: a chevron is a 16px glyph, and making it the control would put a
 * 16px target in a 44px row. No press transform either, unlike an action
 * button: a row is a surface, and a surface that shrinks under the finger reads
 * as a misfire.
 *
 * Controlled, so the page decides what is open. It renders as **two** children
 * of a `SettingsGroup` rather than one, so the group's own divider draws the
 * line between the row and the panel it opened.
 *
 * The panel is always in the DOM and toggles `hidden`, so `aria-controls`
 * always names an element that exists.
 *
 * Not to be confused with `navigation/Disclosure`, which is a standalone
 * heading that opens; this one is a settings row.
 */
const meta = {
  title: "Design system/Navigation/DisclosureRow",
  component: DisclosureRow,
  args: {
    label: "Per-workspace overrides",
    isOpen: false,
    onToggle: () => {},
    children: null,
  },
  parameters: { layout: "padded" },
} satisfies Meta<typeof DisclosureRow>

export default meta

type Story = StoryObj<typeof meta>

/** Press the row to open it. The chevron rotates on the 150ms rung, guarded. */
export const Default: Story = {
  render: () => {
    const [open, setOpen] = useState(false)
    return (
      <SettingsGroup bounded title="Code execution">
        <DisclosureRow
          label="Per-workspace overrides"
          help="Workspaces that do not follow the deployment default."
          trailing={<span className="text-caption text-subtle">2</span>}
          isOpen={open}
          onToggle={() => setOpen(!open)}
        >
          <SettingRow
            nested
            label="checkout-service"
            help="Network access allowed."
            control={
              <Toggle label="checkout-service" isSelected onChange={() => {}} />
            }
          />
          <SettingRow
            nested
            label="batch-ingest"
            help="Code execution refused."
            control={
              <Toggle
                label="batch-ingest"
                isSelected={false}
                onChange={() => {}}
              />
            }
          />
        </DisclosureRow>
      </SettingsGroup>
    )
  },
}

/** Open, so the panel and its nested rows are visible without a press. */
export const Open: Story = {
  render: () => (
    <SettingsGroup bounded title="Code execution">
      <DisclosureRow
        label="Per-workspace overrides"
        help="Workspaces that do not follow the deployment default."
        trailing={<span className="text-caption text-subtle">1</span>}
        isOpen
        onToggle={() => {}}
      >
        <SettingRow
          nested
          label="checkout-service"
          help="Network access allowed."
          control={
            <Toggle label="checkout-service" isSelected onChange={() => {}} />
          }
        />
      </DisclosureRow>
    </SettingsGroup>
  ),
}

/** Without `help` or `trailing`, which is the least a row can carry. */
export const LabelOnly: Story = {
  render: () => {
    const [open, setOpen] = useState(false)
    return (
      <SettingsGroup bounded title="Advanced">
        <DisclosureRow
          label="Request headers"
          isOpen={open}
          onToggle={() => setOpen(!open)}
        >
          <SettingRow
            nested
            label="Forward the caller's user agent"
            control={
              <Toggle
                label="Forward the caller's user agent"
                isSelected={false}
                onChange={() => {}}
              />
            }
          />
        </DisclosureRow>
      </SettingsGroup>
    )
  },
}
