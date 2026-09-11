import { type ReactNode, useEffect, useRef, useState } from "react"

import { copyToClipboard } from "../helpers/clipboard"

/**
 * A verbatim block to be read and taken: a label row on the code surface, and
 * the code itself under it.
 *
 * The one place in this product where an edge is right. Everything else is
 * divided by a hairline on the page's own ground, but a code block is a small
 * object sitting on that ground rather than a division of it: it is a thing to
 * be copied whole, and the border is what says where it ends.
 *
 * The copy affordance is a cell of the label row rather than a button floating
 * over the code, which is how the copy family behaves everywhere else here and
 * is also what keeps it off the first line of the snippet.
 *
 * `value` is the exact text a copy yields, which is not always what is
 * rendered: the bundled guide hands this the raw fence while `children` carries
 * react-markdown's own nodes. With no `value` there is nothing to copy and the
 * control is absent rather than dead.
 *
 * With no `label`, the row reads "code" and the block is named "Code": an
 * unlabeled row would be a bar with a control and no subject, and "code" is the
 * honest name for a block whose author did not say what it is.
 */
export function CodeBlock({
  label,
  value,
  children,
  className,
  isBounded = false,
}: {
  /** The left of the label row. A language, usually. */
  label?: string
  /** The text a copy yields. Absent means no copy control. */
  value?: string
  /** Defaults to `value`; pass children when the rendered form differs. */
  children?: ReactNode
  /** Layout only, on the outer block. */
  className?: string
  /**
   * Whether the code scrolls inside a capped block instead of growing the page.
   *
   * Off by default, which is right on a page: prose reads top to bottom and a
   * fence that scrolled would hide lines from anyone skimming. On inside a
   * frame, where a snippet of unknown length would otherwise push whatever sits
   * under it off the bottom, and what sits under it is the thing the frame is
   * about.
   */
  isBounded?: boolean
}) {
  const [copied, setCopied] = useState(false)
  const resetTimer = useRef<ReturnType<typeof setTimeout> | undefined>(
    undefined,
  )
  useEffect(() => () => clearTimeout(resetTimer.current), [])

  const copy = async () => {
    if (value === undefined) return
    // Through the shared helper, which falls back to `execCommand`: this
    // dashboard is routinely served from a non-secure origin, where the async
    // Clipboard API does not exist at all. "Copied" is only claimed when it was.
    if (!(await copyToClipboard(value))) return
    setCopied(true)
    clearTimeout(resetTimer.current)
    resetTimer.current = setTimeout(() => setCopied(false), 2_000)
  }

  return (
    <div className={className}>
      <div className="border-code-border bg-code-control text-mono-micro text-code-foreground flex items-center justify-between gap-3 border px-4 py-1.5">
        <span>{label ?? "code"}</span>
        {value !== undefined ? (
          // The 44px target is bled outward with a pseudo-element rather than
          // taken as height: the label row is 1.5 padding by design, and a
          // control that grew it would make every code block taller.
          <button
            type="button"
            onClick={copy}
            className="relative opacity-75 before:absolute before:-inset-x-2 before:-inset-y-[0.6875rem] before:content-[''] hover:opacity-100"
          >
            {copied ? "Copied" : "Copy"}
          </button>
        ) : null}
      </div>
      {/* biome-ignore-start lint/a11y/noNoninteractiveTabindex: the block scrolls, so it has to be reachable by keyboard */}
      {/* biome-ignore-start lint/a11y/useSemanticElements: the region role is what names the scrollable block for AT */}
      <pre
        tabIndex={0}
        role="region"
        aria-label={label ? `${label} code` : "Code"}
        // `border-t-0` because the label row above already drew that edge.
        className={`border-code-border bg-code-surface text-mono-caption text-code-foreground overflow-x-auto border border-t-0 px-4 py-3.5 leading-5 ${
          isBounded ? "max-h-56 overflow-y-auto" : ""
        }`}
      >
        {children ?? value}
      </pre>
      {/* biome-ignore-end lint/a11y/noNoninteractiveTabindex: see above */}
      {/* biome-ignore-end lint/a11y/useSemanticElements: see above */}
    </div>
  )
}
