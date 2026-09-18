import type { ComponentPropsWithoutRef, ReactElement, ReactNode } from "react"
import { Children, isValidElement } from "react"
import type { Components, ExtraProps } from "react-markdown"
import { CodeBlock } from "@/design-system/content/CodeBlock"
import { Markdown } from "@/design-system/content/Markdown"

import { PageIntro } from "@/design-system/layout/PageIntro"
import { Section } from "@/design-system/layout/Section"
// The operator user guide is bundled straight from the repo's docs so the
// running dashboard ships the guide that matches it, instead of pointing at a
// docs site that may describe a different version. Rebuilding the dashboard
// after editing the guide is what keeps them aligned (see AGENTS.md).
import { welcomeGuideHref } from "@/shared/helpers/welcomeGuide"
import { useDeployment } from "@/shared/hooks/useDeployment"
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
  // Fence state carries from line to line, so the search toggles as it walks and
  // reads a heading only from outside a fenced block. `findIndex` stops at the
  // first hit, which is the same ground the scan needs to cover.
  let isInFence = false
  const offset = lines.slice(start + 1).findIndex((line) => {
    if (/^\s*```/.test(line)) {
      isInFence = !isInFence
      return false
    }
    return !isInFence && /^## /.test(line)
  })
  const end = offset === -1 ? lines.length : start + 1 + offset
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
//      on GitHub and the /welcome tutorial. The remaining sections cover setup,
//      navigation, and operating the dashboard after sign-in.
const guideBody = dropSection(
  dashboardGuide.replace(/^#{1,6}[ \t]+.*\r?\n+/, ""),
  "First-run walkthrough",
)

// react-markdown passes each node's hast `node` to custom components; it must be
// destructured out of the DOM spread or it renders as node="[object Object]" on
// every element (and stringifies the subtree per render). ExtraProps types it.
type MdProps<E extends "a" | "table" | "pre"> = ComponentPropsWithoutRef<E> &
  ExtraProps

export const markdownComponents: Components = {
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
    // biome-ignore lint/a11y/useSemanticElements: <section> would not make the overflow keyboard-reachable, which is the point
    <div
      className="my-4 max-w-full overflow-x-auto"
      // biome-ignore lint/a11y/noNoninteractiveTabindex: a scrollable region must be focusable (axe scrollable-region-focusable)
      tabIndex={0}
      role="region"
      aria-label="Table"
    >
      <table {...props} />
    </div>
  ),
  // Code blocks scroll horizontally on overflow; make them focusable too so the
  // clipped content is keyboard-reachable (axe scrollable-region-focusable).
  pre: (props: MdProps<"pre">) => <MarkdownCodeBlock {...props} />,
}

/**
 * The bundled guide's fenced blocks, rendered through the shared `CodeBlock`.
 *
 * The language comes from the `language-*` class remark puts on the inner
 * `<code>`, and the text a copy yields is that child's own string rather than
 * the rendered nodes, which is why the block takes both: `children` is what
 * react-markdown produced, `value` is what goes on the clipboard.
 */
function MarkdownCodeBlock({ node: _node, children }: MdProps<"pre">) {
  const child = Children.toArray(children).find(isValidElement) as
    | ReactElement<{ className?: string; children?: ReactNode }>
    | undefined
  const language =
    /language-([\w+-]+)/.exec(child?.props.className ?? "")?.[1] ?? ""
  const text =
    typeof child?.props.children === "string" ? child.props.children : ""

  return (
    <CodeBlock
      label={language || undefined}
      value={text || undefined}
      className="my-5"
    >
      {children}
    </CodeBlock>
  )
}

export function DocsPage() {
  // The pointer at the get-started walkthrough is only true where the gateway
  // serving this dashboard serves that page too; see `welcomeGuideHref`.
  const welcomeHref = welcomeGuideHref(useDeployment())

  return (
    <div className="flex flex-col">
      <PageIntro title="User guide">
        A reference for operating this dashboard, bundled with and
        version-matched to the running gateway.
        {welcomeHref ? (
          <>
            {" "}
            New here? The get-started walkthrough lives at{" "}
            {/* A new tab, like the other links to this page: it leaves the SPA,
                and the guide is what the reader came here to keep. */}
            <a
              href={welcomeHref}
              target="_blank"
              rel="noreferrer"
              className="text-link hover:text-link-hover"
            >
              {welcomeHref}
            </a>
            .
          </>
        ) : null}
      </PageIntro>
      {/* One band of the page, like every other surface: the rule runs the
          width of the scroll area and the guide fills the page column rather
          than a measure of its own, so the guide's tables and headings line up
          with the rest of the product on the window it is read on. The type is
          16/26 rather than the 14px everything else uses, because this is read
          rather than scanned. */}
      <Section className="border-t border-border py-8">
        <Markdown components={markdownComponents}>{guideBody}</Markdown>
      </Section>
    </div>
  )
}
