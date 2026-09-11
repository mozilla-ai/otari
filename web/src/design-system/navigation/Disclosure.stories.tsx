import type { Meta, StoryObj } from "@storybook/react-vite"

import { Disclosure } from "./Disclosure"

/**
 * A heading that opens to reveal what is under it.
 *
 * Right for detail an operator asks for (a request's raw payload, a provider's
 * full error), and wrong for anything they need in order to decide: content
 * behind a click is content most people never read, so collapsing something is
 * a decision that it is optional.
 *
 * The chevron travels on the 150ms rung and is guarded, like the rail rows it
 * shares that duration with.
 */
const meta = {
  title: "Design system/Navigation/Disclosure",
  component: Disclosure,
  args: { heading: "Raw request", children: null },
  parameters: { layout: "padded" },
} satisfies Meta<typeof Disclosure>

export default meta

type Story = StoryObj<typeof meta>

export const Default: Story = {
  render: () => (
    <div className="w-96">
      <Disclosure heading="Raw request">
        <pre className="overflow-x-auto font-mono text-mono-caption">
          {'{\n  "model": "gpt-4o-mini",\n  "stream": true\n}'}
        </pre>
      </Disclosure>
    </div>
  ),
}

export const DefaultExpanded: Story = {
  render: () => (
    <div className="w-96">
      <Disclosure heading="Why this request was refused" isDefaultExpanded>
        <p className="text-body">
          The workspace budget was exhausted 4 minutes before this request
          arrived.
        </p>
      </Disclosure>
    </div>
  ),
}

/**
 * Stacked, divided by the rules each row draws. Each is independent: this is a
 * disclosure rather than an accordion, so opening one does not close another.
 */
export const Stacked: Story = {
  render: () => (
    <div className="flex w-96 flex-col divide-y divide-border border-y border-border">
      {[
        ["Raw request", "The body this gateway received."],
        ["Provider response", "What the upstream returned, before sanitizing."],
        [
          "Routing decision",
          "Which candidates were tried, and why each failed.",
        ],
      ].map(([heading, body]) => (
        <Disclosure key={heading} heading={heading}>
          <p className="text-body">{body}</p>
        </Disclosure>
      ))}
    </div>
  ),
}
