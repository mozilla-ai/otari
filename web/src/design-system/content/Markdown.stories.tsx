import type { Meta, StoryObj } from "@storybook/react-vite"

import { Markdown } from "./Markdown"

const SAMPLE = `# Operating this gateway

A reference for the dashboard, bundled with and version-matched to the running
gateway.

## Keys and budgets

A key authenticates a caller. Each one carries its own budget and model scope,
and a request is refused when either is exhausted.

### Creating one

Run \`otari gen-secret-key\` and store the result; it is shown once.

1. Add a provider credential.
2. Create an API key.
3. Point a client at this gateway's base URL.
   - The base URL is the origin this dashboard is served from.
   - Hosted deployments use the data-plane URL instead.

> A budget is enforced at reservation time, not at settlement, so a burst
> cannot overshoot it.

| Setting | Default | Effect |
| --- | --- | --- |
| \`require_pricing\` | on | Refuses a request to an unpriced model |
| \`model_cache_ttl_seconds\` | 300 | How long discovery is reused |

See [the configuration reference](https://example.com/config) for the rest.

---

Anything not listed above is startup-only.
`

/**
 * Rendered Markdown, on the design tokens.
 *
 * The component exists to **own the DOM**. Styling generated markup used to
 * mean about thirty descendant selectors on an `.otari-markdown` wrapper
 * (`.otari-markdown p`, `.otari-markdown li > ul`, `.otari-markdown th`), which
 * is a wrapper class that restyles everything beneath it wherever it appears.
 * react-markdown takes a `components` map, so every element here is one of ours
 * wearing its own utilities.
 *
 * Two heading steps and no more: the display face at 20px for a section, then
 * `text-title` for anything under it. `h3`/`h4` had been hand-spelling exactly
 * that role's values.
 */
const meta = {
  title: "Design system/Content/Markdown",
  component: Markdown,
  args: { children: SAMPLE },
  parameters: { layout: "padded" },
} satisfies Meta<typeof Markdown>

export default meta

type Story = StoryObj<typeof meta>

/** Every element the prose set covers, at the 560px measure the guide uses. */
export const Default: Story = { args: { className: "max-w-[35rem]" } }

/** The same document on the dark artboard; every value resolves through tokens. */
export const Dark: Story = {
  args: { className: "max-w-[35rem]" },
  globals: { theme: "dark" },
}

/**
 * `components` merges over the prose set, for an element whose behavior belongs
 * to the caller. This is how `DocsPage` supplies a link that rewrites its href
 * against the repository, a table wrapped in a focusable scroller, and a code
 * block with a copy control: presentation stays in the component, anything that
 * has to know about the application arrives through this prop.
 */
export const CallerOverride: Story = {
  args: {
    className: "max-w-[35rem]",
    children:
      "A [link](https://example.com) rendered by the caller's own renderer.\n",
    components: {
      a: ({ node: _node, children, ...props }) => (
        <a
          {...props}
          target="_blank"
          rel="noreferrer"
          className="text-danger underline"
        >
          {children} ↗
        </a>
      ),
    },
  },
}

/** The narrow case: the measure is the caller's, so it wraps rather than clipping. */
export const Narrow: Story = { args: { className: "max-w-[20rem]" } }
