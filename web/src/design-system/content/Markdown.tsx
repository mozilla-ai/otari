import type { ComponentPropsWithoutRef, ReactNode } from "react"
import ReactMarkdown, { type Components, type ExtraProps } from "react-markdown"
import remarkGfm from "remark-gfm"

/**
 * Rendered Markdown, on the design tokens.
 *
 * **It exists to own the DOM.** Markdown is generated markup, so styling it
 * used to mean a block of descendant selectors keyed on a wrapper class
 * (`.otari-markdown p`, `.otari-markdown li > ul`, and thirty more). That is
 * the bleed this component removes: react-markdown takes a `components` map, so
 * every element can be one of ours wearing its own utilities, and a rule that
 * lives on the element cannot reach anything else on the page.
 *
 * It also fixed a drift the old shape could not prevent. The headings were set
 * with `@apply text-display-sub`, and globals.css's own comment records why:
 * "Generated markup carries no class, so this reaches the role with `@apply`
 * rather than by wearing it" -- and that those values "had already stopped
 * tracking the scale once". Here the heading *wears* the role, and `h3`/`h4`
 * turned out to be `text-title` exactly (16/24 at 550), which they had been
 * hand-spelling.
 *
 * Vertical rhythm is margins rather than a `gap` on the parent, which is the one
 * place this tree's spacing rule does not apply: prose spacing differs per
 * element (a heading owes more room above than a paragraph) and the parent
 * cannot know what its generated children are.
 */

/** react-markdown hands each renderer its hast node; it must not reach the DOM. */
type MdProps<E extends keyof React.JSX.IntrinsicElements> =
  ComponentPropsWithoutRef<E> & ExtraProps

/**
 * A renderer that drops `node` and applies a class.
 *
 * `node` is destructured out on purpose rather than spread: react-markdown
 * passes it to every custom component, and spreading it renders
 * `node="[object Object]"` on the element *and* stringifies the subtree on
 * every render. That is a real defect the docs page hit once already, which is
 * why this is one helper rather than the same destructure written fifteen times.
 */
function proseElement<E extends keyof React.JSX.IntrinsicElements>(
  Tag: E,
  className: string,
) {
  return function ProseElement({ node: _node, ...props }: MdProps<E>) {
    const Element = Tag as "div"
    return <Element className={className} {...(props as object)} />
  }
}

/**
 * The prose elements, each wearing a role or a token utility.
 *
 * Two heading steps and no more, which is a deliberate limit rather than an
 * omission: the display face at 20px for a section, then the title role for
 * anything under it. A guide with four heading sizes spends the whole scale on
 * one document and leaves a reader guessing which of the middle two outranks
 * the other, so `h1`/`h2` and `h3`/`h4` each collapse to one step.
 */
const PROSE: Components = {
  h1: proseElement("h1", "text-display-sub mt-8 mb-3"),
  h2: proseElement("h2", "text-display-sub mt-8 mb-3"),
  h3: proseElement("h3", "text-title mt-7 mb-2"),
  h4: proseElement("h4", "text-title mt-7 mb-2"),
  p: proseElement("p", "my-3"),
  // Thicken the underline on hover rather than lightening the text:
  // `--color-link` is the role tuned to clear 4.5:1 in both themes, and a hover
  // tint that lifted it would trade a contrast guarantee for a hover cue.
  a: proseElement(
    "a",
    "text-link underline underline-offset-2 hover:decoration-2",
  ),
  ul: proseElement("ul", "my-3 list-disc pl-6 [&_ul]:my-1 [&_ol]:my-1"),
  ol: proseElement("ol", "my-3 list-decimal pl-6 [&_ul]:my-1 [&_ol]:my-1"),
  li: proseElement("li", "my-1"),
  // The face is the marker, and nothing else is. This was a tinted chip, which
  // made it the last place in the product where the accent was decoration
  // rather than data ink. Fira Code beside the body face is already an
  // unmistakable change of voice, which is how an identifier is marked in every
  // table here; a fill on top of it said the same thing twice. No size
  // reduction either: `0.85em` shrank an identifier a reader has to copy by eye
  // below the prose around it, and mono already reads smaller at a matched size.
  code: proseElement("code", "font-mono"),
  blockquote: proseElement(
    "blockquote",
    "my-4 border-l-[0.1875rem] border-accent py-1 pl-4 text-muted",
  ),
  hr: proseElement("hr", "my-6 border-0 border-t border-border"),
  // The same table the rest of the product has: rows on the page ground divided
  // by rules, with the header told from them by a rule and by weight rather
  // than by a fill.
  table: proseElement("table", "w-full border-collapse text-body"),
  th: proseElement(
    "th",
    "border-0 border-b border-border py-2 pr-3 text-left align-top font-semibold text-foreground",
  ),
  td: proseElement(
    "td",
    "border-0 border-b border-border-subtle py-2 pr-3 text-left align-top",
  ),
}

export function Markdown({
  children,
  components,
  className = "",
}: {
  /** The Markdown source. */
  children: string
  /**
   * Renderers to merge over the prose set, for an element whose behavior
   * belongs to the caller: a link that rewrites its href against a docs base,
   * a code block with a copy control, a table that needs a scroll wrapper.
   * Presentation stays here; anything that has to know about the application
   * arrives through this.
   */
  components?: Components
  /** Layout and measure at the call site. Not for restyling the prose. */
  className?: string
}): ReactNode {
  return (
    // 16px on a 26px line, and the step *up* from the product's 14px is the
    // point: everything else in this dashboard is scanned and this is read.
    // Off the type scale deliberately, and the only place that is true, which
    // is why it is spelled once here rather than as a role nothing else wants.
    //
    // `[&>*:first-child]:mt-0` replaces `.otari-markdown > :first-child`: the
    // first block's own top margin would otherwise push the whole document down
    // from whatever sits above it. Scoped to this element's direct children, so
    // unlike the old selector it cannot reach a nested document.
    <div
      className={`text-base leading-[1.625rem] text-foreground [&>*:first-child]:mt-0 ${className}`}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{ ...PROSE, ...components }}
      >
        {children}
      </ReactMarkdown>
    </div>
  )
}
