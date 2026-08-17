import { Card } from "@heroui/react"
import type { ComponentPropsWithoutRef } from "react"
import type { Components, ExtraProps } from "react-markdown"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"

import { PageHeader } from "@/shared/components/ui"
// The operator user guide is bundled straight from the repo's docs so the
// running dashboard ships the guide that matches it, instead of pointing at a
// docs site that may describe a different version. Rebuilding the dashboard
// after editing the guide is what keeps them aligned (see AGENTS.md).
import dashboardGuide from "../../../../docs/dashboard.md?raw"

// The bundled guide lives among sibling docs (configuration.md, quickstart.md,
// ...) that are not bundled here, so its relative links cannot resolve inside
// the SPA. Point them at the rendered source on GitHub instead.
const DOCS_SOURCE_BASE = "https://github.com/mozilla-ai/otari/blob/main/docs/"
// The bundled guide is dashboard.md, so an in-page "#frag" link targets a
// heading in that same file's source.
const SELF_DOC = "dashboard.md"

function resolveDocHref(href: string | undefined): string | undefined {
  if (!href) return href
  // Absolute (scheme:...) or protocol-relative URLs pass through untouched.
  if (/^[a-z][a-z0-9+.-]*:/i.test(href) || href.startsWith("//")) {
    return href
  }
  // In-page anchors ("#the-two-key-model", or remark-gfm's "#user-content-fn-*"
  // footnotes) cannot resolve here: hash routing owns location.hash, so following
  // one rewrites the route and the "*" fallback bounces the operator to
  // Overview. Send them to the rendered source on GitHub, like sibling docs.
  if (href.startsWith("#")) {
    return DOCS_SOURCE_BASE + SELF_DOC + href
  }
  return DOCS_SOURCE_BASE + href.replace(/^\.\//, "")
}

function isExternal(href: string | undefined): boolean {
  return !!href && (/^https?:\/\//i.test(href) || href.startsWith("//"))
}

// Drop a whole "## <title>" section (through to the next level-2 heading) from
// the guide, skipping any "## " that appears inside a fenced code block.
function dropSection(md: string, title: string): string {
  const lines = md.split("\n")
  const start = lines.findIndex((line) => line.trim() === `## ${title}`)
  if (start === -1) return md
  let end = lines.length
  let inFence = false
  for (let i = start + 1; i < lines.length; i += 1) {
    if (/^\s*```/.test(lines[i])) {
      inFence = !inFence
    } else if (!inFence && /^## /.test(lines[i])) {
      end = i
      break
    }
  }
  // Collapse the blank-line run left where the section was removed.
  return [...lines.slice(0, start), ...lines.slice(end)]
    .join("\n")
    .replace(/\n{3,}/g, "\n\n")
}

// Two edits turn the general guide into an in-dashboard reference:
//   1. The guide opens with its own top-level title ("# Admin dashboard"), which
//      would duplicate the page header below. Drop that leading ATX heading so
//      the page shows a single title; the intro paragraph then flows under it.
//      Anchored to an ATX heading (`#` + space) so a first line that merely
//      starts with `#` is left alone.
//   2. The reader reached this page through a running, signed-in dashboard, so
//      the first-run walkthrough (start the gateway, find the master key, sign
//      in) is circular here. Drop it; it stays in docs/dashboard.md for readers
//      on GitHub and the /welcome tutorial, and its post-sign-in substance is
//      covered by the page-by-page reference below.
const guideBody = dropSection(
  dashboardGuide.replace(/^#{1,6}[ \t]+.*\r?\n+/, ""),
  "First-run walkthrough",
)

// react-markdown passes each node's hast `node` to custom components; it must be
// destructured out of the DOM spread or it renders as node="[object Object]" on
// every element (and stringifies the subtree per render). ExtraProps types it.
type MdProps<E extends "a" | "table" | "pre"> = ComponentPropsWithoutRef<E> &
  ExtraProps

const markdownComponents: Components = {
  a: ({ node: _node, href, children, ...props }: MdProps<"a">) => {
    const resolved = resolveDocHref(href)
    // Every rewritten link is now an absolute GitHub URL (external), so it opens
    // in a new tab and the operator does not lose the dashboard.
    const external = isExternal(resolved)
    return (
      <a
        href={resolved}
        {...(external ? { target: "_blank", rel: "noreferrer" } : {})}
        {...props}
      >
        {children}
      </a>
    )
  },
  // The two-key table is wider than a narrow viewport. Keep the native <table>
  // display so it retains its table/row/cell a11y semantics (a display:block
  // table drops role=table in WebKit), and put the horizontal scroll on a
  // focusable wrapper so keyboard users can reach the clipped columns.
  table: ({ node: _node, ...props }: MdProps<"table">) => (
    <div
      className="otari-markdown-scroll"
      tabIndex={0}
      role="region"
      aria-label="Table"
    >
      <table {...props} />
    </div>
  ),
  // Code blocks scroll horizontally on overflow; make them focusable too so the
  // clipped content is keyboard-reachable (axe scrollable-region-focusable).
  pre: ({ node: _node, ...props }: MdProps<"pre">) => (
    <pre tabIndex={0} role="region" aria-label="Code" {...props} />
  ),
}

export function DocsPage() {
  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="User guide"
        description="A reference for operating this dashboard, bundled with and version-matched to the running gateway. New here? The get-started walkthrough lives at /welcome."
      />
      <Card>
        <Card.Content className="p-5 sm:p-6">
          <div className="otari-markdown">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={markdownComponents}
            >
              {guideBody}
            </ReactMarkdown>
          </div>
        </Card.Content>
      </Card>
    </div>
  )
}
