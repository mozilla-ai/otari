import type { Meta, StoryObj } from "@storybook/react-vite"
import { useState } from "react"

import { Button } from "../actions/Button"
import { Field } from "../forms/Field"
import { INPUT_CLASS } from "../forms/inputClass"
import { Toggle } from "../forms/Toggle"
import { FilterSelect } from "../navigation/FilterSelect"
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
 * **The lane is one width down the page and the control fills it.** Sized per
 * control it was not a lane at all, which `SharedLane` below is about.
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

/**
 * Four different controls, one lane.
 *
 * This is the story worth looking at, because it is the one that used to be
 * wrong: a text field, a select and a two-digit number each sized themselves,
 * so a page of rows had a different left edge on every one of them. The lane is
 * a fixed-width slot now, the same rule `design/layout.md` states for any
 * repeated row, and a control fills it: `w-full` on a field, `fullWidth` on a
 * `FilterSelect`. A control that cannot fill a lane, a toggle, sits at its
 * leading edge rather than floating to the row's end.
 */
export const SharedLane: Story = {
  render: () => {
    const [on, setOn] = useState(true)
    const [mode, setMode] = useState("default")
    return (
      <SettingsGroup bounded title="Web search">
        <SettingRow
          label="Backend URL"
          configKey="web_search_url"
          help="While unset, otari_web_search requests are rejected with 400."
          control={
            // The shape the lane exists for: a field beside its own button. It
            // takes `min-w-0 flex-1` rather than `w-full` so the button keeps
            // its size and the field gives up the difference.
            <div className="flex items-center gap-1.5">
              <input
                aria-label="Backend URL"
                defaultValue="http://searxng:8080"
                className={`otari-machine-field min-w-0 flex-1 ${INPUT_CLASS}`}
              />
              <Button size="sm" className="min-w-[5.5rem] shrink-0">
                Test
              </Button>
            </div>
          }
        />
        <SettingRow
          label="Max results"
          configKey="web_search_max_results"
          help="Cap on hits per call. A per-tool max_results still overrides it."
          control={
            <input
              aria-label="Max results"
              defaultValue="10"
              className={`otari-machine-field w-full text-right tabular-nums ${INPUT_CLASS}`}
            />
          }
        />
        <SettingRow
          label="Extract page content"
          configKey="web_search_extract"
          help="On: page text is extracted in-process. Off: snippet-only results."
          control={
            <FilterSelect
              fullWidth
              ariaLabel="Extract page content"
              value={mode}
              onChange={setMode}
              options={[
                { value: "default", label: "Default (on)" },
                { value: "on", label: "On" },
                { value: "off", label: "Off" },
              ]}
            />
          }
        />
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
