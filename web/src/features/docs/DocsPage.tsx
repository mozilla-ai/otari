import type { ComponentPropsWithoutRef, ReactElement, ReactNode } from "react"
import { Children, isValidElement, useEffect, useRef, useState } from "react"
import type { Components, ExtraProps } from "react-markdown"
import { Markdown } from "@/design-system/content/Markdown"

import { PageIntro } from "@/design-system/layout/PageIntro"
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
  pre: (props: MdProps<"pre">) => <CodeBlock {...props} />,
}

/**
 * A code block with a label row above it: the language on the left, a copy
 * affordance on the right.
 *
 * The row is a cell of the block rather than a button floating on it, which is
 * how the copy family works everywhere else in this product and is also what
 * keeps the control out of the text it would otherwise sit over. The language
 * comes from the `language-*` class remark puts on the inner `<code>`, and the
 * row is dropped entirely when there is neither a language nor anything to
 * copy, rather than rendering an empty bar.
 */
function CodeBlock({ node: _node, children, ...props }: MdProps<"pre">) {
  const [copied, setCopied] = useState(false)
  const resetTimer = useRef<ReturnType<typeof setTimeout> | undefined>(
    undefined,
  )
  useEffect(() => () => clearTimeout(resetTimer.current), [])

  const child = Children.toArray(children).find(isValidElement) as
    | ReactElement<{ className?: string; children?: ReactNode }>
    | undefined
  const language =
    /language-([\w+-]+)/.exec(child?.props.className ?? "")?.[1] ?? ""
  const text =
    typeof child?.props.children === "string" ? child.props.children : ""

  const copy = async () => {
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text)
        setCopied(true)
        clearTimeout(resetTimer.current)
        resetTimer.current = setTimeout(() => setCopied(false), 2_000)
      }
    } catch {
      // No Clipboard API, or it refused. The block is selectable either way, so
      // there is nothing to fall back to and nothing to claim.
    }
  }

  return (
    <div className="my-5">
      {/* The label row: the language on the left, the copy affordance on the
          right, both in mono at the caption step on the block's own surface.
          The row is a cell of the block rather than a button floating on it,
          which is how the copy family works everywhere else here. */}
      <div className="flex items-center justify-between gap-3 border border-code-border bg-code-control px-4 py-1.5 text-mono-micro text-code-foreground">
        <span>{language || "code"}</span>
        {text ? (
          <button
            type="button"
            onClick={copy}
            className="opacity-75 hover:opacity-100"
          >
            {copied ? "Copied" : "Copy"}
          </button>
        ) : null}
      </div>
      {/* biome-ignore-start lint/a11y/noNoninteractiveTabindex: same as the table above; the block scrolls, so it has to be reachable */}
      {/* biome-ignore-start lint/a11y/useSemanticElements: the region role is what names the scrollable block for AT */}
      {/* A code block is a small object of its own on the prose ground, which
          is the one place in this product where an edge is right: it is a
          verbatim thing to be copied, not a division of the page. `border-t-0`
          because the label row above already drew that edge. */}
      <pre
        tabIndex={0}
        role="region"
        aria-label={language ? `${language} code` : "Code"}
        className="overflow-x-auto border border-code-border border-t-0 bg-code-surface px-4 py-3.5 text-mono-caption leading-5 text-code-foreground"
        {...props}
      >
        {children}
      </pre>
      {/* biome-ignore-end lint/a11y/noNoninteractiveTabindex: see above */}
      {/* biome-ignore-end lint/a11y/useSemanticElements: see above */}
    </div>
  )
}

export function DocsPage() {
  return (
    <div className="flex flex-col">
      {/* The prose measure, not the app's 620px default: on the one page whose
          subject is the measure, the widest line should not be the scanning-size
          paragraph at the top of it. */}
      <PageIntro title="User guide" descriptionClassName="max-w-[560px]">
        A reference for operating this dashboard, bundled with and
        version-matched to the running gateway. New here? The get-started
        walkthrough lives at /welcome.
      </PageIntro>
      {/* The prose pattern: a 560px measure at 16px, bounded above by the
          section rule and on its right by a rule that runs the height of the
          page, with the ground beyond it left free. The interim 620px cap this
          replaces was a number chosen on this page; 560 at 16/26 is the measure
          the pattern sets, and the type steps *up* from the 14px the rest of
          the product uses, because this is read rather than scanned. */}
      <div className="flex flex-1 border-t border-border">
        <div className="min-w-0 border-r border-border px-4 py-8 md:px-6">
          <Markdown className="max-w-[560px]" components={markdownComponents}>
            {guideBody}
          </Markdown>
        </div>
      </div>
    </div>
  )
}
