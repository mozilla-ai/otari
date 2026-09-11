import type { Meta, StoryObj } from "@storybook/react-vite"
import { Button } from "../actions/Button"
import { Field } from "../forms/Field"
import { Toggle } from "../forms/Toggle"
import { SettingRow } from "./SettingRow"
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

/**
 * `bounded` is the other shape: the heading sits unruled above a framed block of
 * rows, inside the page column rather than bleeding to the scroll area's edges.
 * It reads as one object on a page that stacks several small groups, where the
 * full-width bands run together into a single striped field.
 *
 * It is also what carries `.otari-settings`, the dense place, so **the controls
 * inside a bounded group are 32px rather than the form height**, and 44px at
 * 16px below `md` where a row's control stacks full width. Compare a row here
 * with one in `Default` above.
 */
export const Bounded: Story = {
  render: () => (
    <div className="flex max-w-2xl flex-col gap-6">
      <SettingsGroup
        bounded
        title="Web search"
        description="Which backend answers a search tool call."
      >
        <SettingRow
          label="Backend URL"
          configKey="web_search_url"
          help="Reachable from the gateway, not from the browser."
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
          configKey="web_search_max_uses"
          control={<Field label="Max uses" value="5" onChange={() => {}} />}
        />
      </SettingsGroup>
    </div>
  ),
}

/**
 * `docsHref` trails the description, for the page of the manual this group is
 * about. It renders a `DocsLink`, so it is always external and always opens in
 * a new tab.
 */
export const WithDocsLink: Story = {
  render: () => (
    <SettingsGroup
      bounded
      title="Code execution"
      description="Whether a request may run code, and where."
      docsHref="https://example.com/docs/tools#code-execution"
    >
      <SettingRow
        label="Allow code execution"
        control={
          <Field label="Allow code execution" value="off" onChange={() => {}} />
        }
      />
    </SettingsGroup>
  ),
}
