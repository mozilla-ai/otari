import type { Meta, StoryObj } from "@storybook/react-vite"

import { Button } from "../actions/Button"
import { Field } from "../forms/Field"
import { Toggle } from "../forms/Toggle"
import { SettingsGroup } from "./SettingsGroup"

/**
 * A titled band of related rows, which is what a settings page is made of.
 *
 * Replaces `deprecated/SettingsSection`, whose only difference was an
 * `export const` arrow and a name that shadowed this one. That module has no
 * call site anywhere and is pinned at zero by `deprecated/deprecated.test.ts`.
 */
const meta = {
  title: "Design system/Layout/SettingsGroup",
  component: SettingsGroup,
  args: { title: "Model discovery", children: null },
  parameters: { layout: "padded" },
} satisfies Meta<typeof SettingsGroup>

export default meta

type Story = StoryObj<typeof meta>

export const Default: Story = {
  render: () => (
    <SettingsGroup
      title="Model discovery"
      description="Whether this gateway asks each provider what it serves, or lists only the models priced by hand."
    >
      <div className="flex items-center justify-between gap-4 py-2">
        <span className="text-body">Refresh on startup</span>
        <Toggle label="Refresh on startup" isSelected onChange={() => {}} />
      </div>
      <div className="flex items-center justify-between gap-4 py-2">
        <span className="text-body">Require a price before routing</span>
        <Toggle
          label="Require a price before routing"
          isSelected={false}
          onChange={() => {}}
        />
      </div>
    </SettingsGroup>
  ),
}

/** `count` where a group's size is worth knowing before it is read. */
export const WithCount: Story = {
  render: () => (
    <SettingsGroup title="Stored provider credentials" count={4}>
      <p className="text-caption">Four providers have a key on this gateway.</p>
    </SettingsGroup>
  ),
}

/**
 * Without a title, which is right when the page's own name already names the
 * group. The rows band keeps its rules; what goes is the heading above it.
 */
export const Untitled: Story = {
  render: () => (
    <SettingsGroup>
      <p className="text-caption">
        The band's rules are still here. The heading is not.
      </p>
    </SettingsGroup>
  ),
}

/**
 * A group that saves takes one `primary` at its own foot. Never a floating
 * page-level Save; layout.md says why.
 */
export const WithSave: Story = {
  render: () => (
    <SettingsGroup
      title="Mail delivery"
      description="Used for invitations and password recovery."
    >
      <div className="flex max-w-md flex-col gap-3 py-2">
        <Field label="SMTP host" value="smtp.example.com" onChange={() => {}} />
        <Field
          label="From address"
          value="otari@example.com"
          onChange={() => {}}
        />
        <div className="flex justify-end">
          <Button variant="primary">Save</Button>
        </div>
      </div>
    </SettingsGroup>
  ),
}
