import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"

import { Field } from "../forms/Field"
import { Toggle } from "../forms/Toggle"
import { SettingRow } from "./SettingRow"
import { SettingsGroup } from "./SettingsGroup"

/**
 * What a setting is on the left, the control that changes it on the right.
 *
 * Shared because a settings page is almost entirely this shape, and spelled by
 * hand the label size, the key caption and the control lane drift between
 * groups on the same page.
 *
 * **The row draws no rules of its own.** `SettingsGroup` divides its children,
 * so a border here would give every seam two lines. That is why the stories
 * below put it inside a group rather than showing it bare.
 *
 * Below `md` the control stops sharing the row and stacks full width under the
 * label, which is also where the label becomes a press target worth having, and
 * why `controlId` exists.
 *
 * Every story passes `bounded`, because every real caller does: that is what
 * puts the rows inside `.otari-settings`, the dense place whose controls are
 * 32px beside their label on a desk and 44px at 16px where they stack on a
 * phone. Without it a row renders at the form height and the story would be
 * showing a size the page never uses.
 */
const meta = {
  title: "Design system/Layout/SettingRow",
  component: SettingRow,
  args: { label: "Require a price before routing", control: null },
  parameters: { layout: "padded" },
} satisfies Meta<typeof SettingRow>

export default meta

type Story = StoryObj<typeof meta>

/** Label, help, and a control. The shape most rows are. */
export const Default: Story = {
  render: () => {
    const [on, setOn] = useState(true)
    return (
      <SettingsGroup bounded title="Model discovery">
        <SettingRow
          label="Require a price before routing"
          help="A request to a model with no price is refused rather than served at an unknown cost."
          control={
            <Toggle
              label="Require a price before routing"
              isSelected={on}
              onChange={setOn}
            />
          }
        />
      </SettingsGroup>
    )
  },
}

/**
 * `configKey` names the key the row writes, as a mono caption beside the label.
 *
 * It also carries the accessible name, which is the part worth knowing: "Backend
 * URL" alone is not unique on a page that configures three services, so the
 * control points at both this label and `<labelId>-key` through
 * `aria-labelledby`, and reads as "Backend URL web_search_url".
 */
export const WithConfigKey: Story = {
  render: () => (
    <SettingsGroup bounded title="Web search">
      <SettingRow
        label="Backend URL"
        labelId="web-search-url"
        configKey="web_search_url"
        controlId="web-search-url-input"
        help="Reachable from the gateway, not from the browser."
        control={
          <Field
            label="Backend URL"
            value="https://search.internal"
            onChange={() => {}}
          />
        }
      />
    </SettingsGroup>
  ),
}

/**
 * The two lines a row may add under its help, and the rule about them: **only an
 * error may add a line**, which is what keeps the help text from rewrapping
 * under the cursor while a value is being changed. `note` is an outcome the row
 * reports back, such as a reachability result.
 */
export const NoteAndError: Story = {
  render: () => (
    <SettingsGroup bounded title="Web search">
      <SettingRow
        label="Backend URL"
        help="Reachable from the gateway, not from the browser."
        note={<p className="text-caption text-success">Reachable, 41ms.</p>}
        control={
          <Field
            label="Backend URL"
            value="https://search.internal"
            onChange={() => {}}
          />
        }
      />
      <SettingRow
        label="Max uses per request"
        help="A ceiling the gateway enforces in every request format."
        error="Must be a whole number between 1 and 20."
        errorId="max-uses-error"
        control={
          <Field label="Max uses" value="0" onChange={() => {}} isInvalid />
        }
      />
    </SettingsGroup>
  ),
}

/**
 * `nested` indents a row, which is how it says it belongs to the row above
 * rather than being its sibling. Used by a disclosure's panel, whose rows are
 * children of the row that opened them.
 */
export const Nested: Story = {
  render: () => {
    const [on, setOn] = useState(true)
    return (
      <SettingsGroup bounded title="Code execution">
        <SettingRow
          label="Allow code execution"
          help="Requests may run code in a sandbox."
          control={
            <Toggle
              label="Allow code execution"
              isSelected={on}
              onChange={setOn}
            />
          }
        />
        <SettingRow
          nested
          label="Allow network access"
          help="The sandbox may reach the internet."
          control={
            <Toggle
              label="Allow network access"
              isSelected={false}
              onChange={() => {}}
            />
          }
        />
        <SettingRow
          nested
          label="Allow image output"
          control={
            <Toggle
              label="Allow image output"
              isSelected={false}
              onChange={() => {}}
            />
          }
        />
      </SettingsGroup>
    )
  },
}
