import type { Meta, StoryObj } from "@storybook/react-vite"

import { CodeBlock } from "./CodeBlock"

const CURL = `curl 'https://gateway.example.com/api/v1/chat/completions' \\
  -H "Otari-Key: gw-3f9a2c5e8b1d4a7f" \\
  -H "Content-Type: application/json" \\
  -d '{"model": "openai:gpt-4o-mini", "messages": [{"role": "user", "content": "Hello"}]}'`

const meta = {
  title: "Design system/Content/CodeBlock",
  component: CodeBlock,
  args: { label: "curl", value: CURL },
} satisfies Meta<typeof CodeBlock>

export default meta

type Story = StoryObj<typeof meta>

/** The language on the left, the copy affordance on the right. */
export const Default: Story = {}

/** No language: the row reads "code" and the block is named "Code". */
export const Unlabeled: Story = {
  args: { label: undefined, value: "uv run otari serve" },
}

/** Nothing to copy, so no control rather than a dead one. */
export const NothingToCopy: Story = {
  args: {
    value: undefined,
    children: "A block whose text this component was never handed.",
  },
}

/** A rendered form that differs from what a copy yields. */
export const RenderedChildren: Story = {
  args: {
    label: "python",
    value: 'client.chat.completions.create(model="openai:gpt-4o-mini")',
    children: (
      <code>
        client.chat.completions.create(model=
        <span className="text-code-foreground">"openai:gpt-4o-mini"</span>)
      </code>
    ),
  },
}

/** Long lines scroll inside the block rather than widening the page. */
export const Overflowing: Story = {
  args: {
    label: "text",
    value: `${"a-very-long-single-token-".repeat(12)}end`,
    className: "max-w-md",
  },
}

/**
 * Bounded: a tall snippet scrolls inside a capped block instead of pushing
 * whatever sits under it off the bottom. For a frame, not for a page.
 */
export const Bounded: Story = {
  args: {
    label: "agent",
    isBounded: true,
    value: Array.from(
      { length: 30 },
      (_, index) => `line ${index + 1} of a prompt nobody wants to scroll past`,
    ).join("\n"),
  },
}
