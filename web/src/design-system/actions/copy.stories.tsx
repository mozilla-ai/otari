import type { Meta, StoryObj } from "@storybook/react-vite"
import { useRef, useState } from "react"

import { Button as ActionButton } from "./Button"
import { CopyButton } from "./CopyButton"
import { CONCEALED_SECRET, CopyableValue, CopyField } from "./CopyField"

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
    value: `curl https://gateway.example.com/api/v1/chat/completions \\
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

/**
 * `concealed` hands a credential over without putting it on screen.
 *
 * The plaintext reaches the DOM only once the operator asks for it, while Copy
 * copies the real value either way. So a key can be pasted elsewhere without
 * ever being read off the screen, which is the point: a screen share or a
 * screenshot of this page does not leak it.
 *
 * `CONCEALED_SECRET` is the stand-in, and it is one fixed run rather than a
 * bullet per character. The length of a key is itself something not to show.
 *
 * A snippet passes the same snippet built around the stand-in, so what is
 * hidden is the key rather than the request that explains it.
 */
export const Concealed: Story = {
  render: () => (
    <div className="flex w-[34rem] flex-col gap-4">
      <CopyField
        label="API key"
        value="sk-otari-4f8a2c9e1b7d3a6f5e0c8b2d"
        concealed={CONCEALED_SECRET}
      />
      <CopyField
        label="Example request"
        multiline
        value={`curl https://gateway.example.com/api/v1/chat/completions \\\n  -H "Authorization: Bearer sk-otari-4f8a2c9e1b7d3a6f5e0c8b2d"`}
        concealed={`curl https://gateway.example.com/api/v1/chat/completions \\\n  -H "Authorization: Bearer ${CONCEALED_SECRET}"`}
      />
    </div>
  ),
}

/**
 * `fieldRef` hands the field's element to the caller, so something outside it
 * can put the caret in the value.
 *
 * The Keys page uses it for the one-time reveal: the key appears and the field
 * is selected, so Ctrl/Cmd-C works without aiming at the button. That matters
 * because the Clipboard API is undefined on the non-secure origins this
 * dashboard is routinely served from, which is the same reason the field
 * selects on click.
 */
/**
 * A field that opens revealed, for the one screen that exists to hand a
 * credential over.
 *
 * `defaultRevealed` is the uncontrolled form of it. The toggle still works and
 * Copy still copies the real value, so nothing otari-ai#2111 asked for is given
 * up; what changes is which state the field starts in. Reach for it only where
 * there is no second chance to read the value.
 */
export const RevealedOnArrival: Story = {
  render: () => (
    <div className="flex w-[34rem] flex-col gap-4">
      <CopyField
        label="Secret key"
        value="otari-sk-9f3a1c77b0e244d1e8a972fc0a3bc5d8"
        concealed={CONCEALED_SECRET}
        defaultRevealed
      />
    </div>
  ),
}

/**
 * Several fields carrying one credential, revealing and concealing together.
 *
 * `isRevealed` and `onRevealChange` make the field controlled, so the caller
 * holds one piece of state and the key and the requests that embed it cannot
 * end up in two states on one screen. The eye stays on every field: whichever
 * one is pressed moves all of them.
 */
export const CoupledReveal: Story = {
  render: function CoupledRevealStory() {
    const [isRevealed, setIsRevealed] = useState(true)
    const key = "otari-sk-9f3a1c77b0e244d1e8a972fc0a3bc5d8"
    const request = (secret: string) =>
      `curl https://gateway.example.com/v1/chat/completions \\\n  -H "Authorization: Bearer ${secret}"`
    return (
      <div className="flex w-[34rem] flex-col gap-4">
        <CopyField
          label="Secret key"
          value={key}
          concealed={CONCEALED_SECRET}
          isRevealed={isRevealed}
          onRevealChange={setIsRevealed}
        />
        <CopyField
          label="Example request"
          multiline
          value={request(key)}
          concealed={request(CONCEALED_SECRET)}
          isRevealed={isRevealed}
          onRevealChange={setIsRevealed}
        />
      </div>
    )
  },
}

export const WithFieldRef: Story = {
  render: () => {
    const fieldRef = useRef<HTMLInputElement | HTMLTextAreaElement | null>(null)
    return (
      <div className="flex w-[34rem] flex-col gap-3">
        <CopyField
          label="API key"
          value="sk-otari-4f8a2c9e1b7d3a6f5e0c8b2d"
          fieldRef={fieldRef}
        />
        <button
          type="button"
          className="min-h-11 self-start border border-control-border px-3 text-body"
          onClick={() => {
            fieldRef.current?.focus()
            fieldRef.current?.select()
          }}
        >
          Select the value from outside the field
        </button>
      </div>
    )
  },
}

/**
 * `action` puts a control beside the field, which moves the copy affordance
 * inside the field and drops the button from the label row. It is the
 * arrangement a value comes in when the operator has something to do with it
 * once pasted: paste the record, then press Verify.
 *
 * Typed `ReactElement` rather than `ReactNode`, because the latter admits
 * `false`: `action={enabled && <Button />}` compiled and then silently fell
 * back to the default arrangement, which is the one this exists to replace.
 * It is also excluded from `multiline` at the type level, since the right
 * padding an in-field control needs indents every line of a textarea instead
 * of making room on the first.
 */
export const FieldWithAction: Story = {
  render: () => (
    <div className="w-[48rem]">
      <CopyField
        label="Copy TXT record for example.com"
        value="otari-verify=9f3c1a7b4e2d8065af13c9b27d4e5f60"
        action={<ActionButton variant="primary">Verify</ActionButton>}
      />
    </div>
  ),
}
