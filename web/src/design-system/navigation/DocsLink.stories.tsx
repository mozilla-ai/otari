import type { Meta, StoryObj } from "@storybook/react-vite"

import { DocsLink } from "./DocsLink"

/**
 * A link out to the documentation, at the end of the sentence it explains.
 *
 * Always external. `href` names a deployment's own documentation site or a
 * document rendered outside this app, never a route the router owns, which is
 * why it is a plain `<a target="_blank">` rather than a TanStack `Link`: an
 * internal route through a bare anchor is a full page reload.
 *
 * The arrow is the external mark, so it is `aria-hidden` and the accessible
 * name stays the word alone.
 */
const meta = {
  title: "Design system/Navigation/DocsLink",
  component: DocsLink,
  args: { href: "https://example.com/docs/tools" },
  parameters: { layout: "padded" },
} satisfies Meta<typeof DocsLink>

export default meta

type Story = StoryObj<typeof meta>

/** "Docs" is the default word, which is what most rows want. */
export const Default: Story = {}

/** A specific word, where "Docs" would not say which document. */
export const NamedDocument: Story = {
  args: { children: "Code execution protocol" },
}

/** Where it actually sits: trailing the help text of a settings row. */
export const InContext: Story = {
  render: () => (
    <p className="max-w-prose text-caption text-subtle">
      Requests are refused when no sandbox backend is reachable.{" "}
      <DocsLink href="https://example.com/docs/tools#per-workspace-code-policy" />
    </p>
  ),
}
