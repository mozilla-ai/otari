import type { Meta, StoryObj } from "@storybook/react-vite"

import { CopyButton } from "./CopyButton"
import { CopyableValue, CopyField } from "./CopyField"

/**
 * The three ways a value an operator has to paste elsewhere is handed over.
 *
 * One file, because choosing between them is one decision: `CopyField` for a
 * value being handed out (a fresh key, a curl snippet), `CopyableValue` for an
 * identifier inside a table row, `CopyButton` when the layout is already yours.
 *
 * All three degrade on a non-secure origin, which this dashboard is routinely
 * served from: the Clipboard API is undefined there, so the text is selected on
 * click and Ctrl/Cmd-C always works. "Copied!" is only claimed when it truly
 * copied, the tooltip says so, and a blocked copy says that instead.
 */
// Required props on the meta, so a story that supplies its own `render` still
// satisfies the component's contract without restating them.
const meta = {
  title: "Design system/Actions/Copy",
  component: CopyField,
  args: {
    label: "Your new API key",
    value: "otari_sk_9f3c1a7b4e2d8065af13c9b27d4e5f60",
  },
} satisfies Meta<typeof CopyField>

export default meta

type Story = StoryObj<typeof meta>

/**
 * The label is a real `<label>` for the field, not a caption beside it: these
 * values are handed over in pairs and threes, so "which field is this" has to be
 * answerable by a screen reader.
 */
export const Field: Story = {
  args: {
    label: "Your new API key",
    value: "otari_sk_9f3c1a7b4e2d8065af13c9b27d4e5f60",
  },
  render: (args) => (
    <div className="w-[36rem]">
      <CopyField {...args} />
    </div>
  ),
}

/** `multiline` swaps the input for a textarea, for a snippet rather than a token. */
export const MultilineField: Story = {
  args: {
    label: "Try it with curl",
    multiline: true,
    value: `curl https://gateway.example.com/v1/chat/completions \\
  -H "Authorization: Bearer otari_sk_…" \\
  -H "Content-Type: application/json" \\
  -d '{"model":"openai:gpt-4o-mini","messages":[{"role":"user","content":"hi"}]}'`,
  },
  render: (args) => (
    <div className="w-[36rem]">
      <CopyField {...args} />
    </div>
  ),
}

/** How the Keys page's one-time reveal actually stacks them. */
export const FieldGroup: Story = {
  render: () => (
    <div className="flex w-[36rem] flex-col gap-4">
      <CopyField label="Key id" value="key_01JQZ8X2M4" />
      <CopyField
        label="Secret"
        value="otari_sk_9f3c1a7b4e2d8065af13c9b27d4e5f60"
      />
    </div>
  ),
}

/**
 * An identifier inside a row, takeable either way: highlighted with the mouse
 * like ordinary text, or copied in one press. The pointer handlers exist so a
 * text drag survives a react-aria table row's own press handling.
 */
export const InlineValue: Story = {
  render: () => (
    <div className="flex w-[32rem] flex-col divide-y divide-border rounded-lg border border-border bg-surface">
      {[
        "req_01JQZ8X2M4B7VYK3",
        "req_01JQZ8X2M4B7VYK4",
        "req_01JQZ8X2M4B7VYK5",
      ].map((id) => (
        <div key={id} className="flex items-center justify-between px-4 py-2">
          <CopyableValue
            value={id}
            label="request id"
            className="font-mono text-caption"
          />
          <span className="text-caption">200</span>
        </div>
      ))}
    </div>
  ),
}

/**
 * `children` renders a display form that differs from what a copy yields: the
 * full model key is copied, the truncated one is shown.
 */
export const InlineValueTruncated: Story = {
  render: () => (
    <CopyableValue
      value="openai:gpt-4o-mini-2024-07-18"
      label="model"
      className="font-mono text-caption"
    >
      openai:gpt-4o-mini…
    </CopyableValue>
  ),
}

/** The button alone, when the surrounding layout is already yours. */
export const Button: Story = {
  render: () => (
    <span className="inline-flex items-center gap-2">
      <span className="font-mono text-caption">
        https://gateway.example.com/v1
      </span>
      <CopyButton value="https://gateway.example.com/v1" label="base URL" />
    </span>
  ),
}
