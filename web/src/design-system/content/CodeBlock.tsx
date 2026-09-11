import { type ReactNode, useEffect, useRef, useState } from "react"
import { FiCheck, FiCopy } from "react-icons/fi"

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
 *
 * `arrangement` picks between the two shapes. `labeled` is the prose default
 * above. `bare` drops the row entirely and floats the copy control on the code
 * itself, for a block whose subject is already named beside it: the setup
 * sheet's tab row says which language this is, so a label row under it would
 * say it twice and cost a band of the frame to do it.
 */
export function CodeBlock({
  label,
  value,
  children,
  className,
  isBounded = false,
  arrangement = "labeled",
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
  /** `labeled` draws the language row; `bare` floats the copy on the code. */
  arrangement?: "labeled" | "bare"
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

  const canCopy = value !== undefined
  const copyLabel = copied ? "Copied" : `Copy ${label ?? "code"}`
  const block = (
    // biome-ignore-start lint/a11y/noNoninteractiveTabindex: the block scrolls, so it has to be reachable by keyboard
    // biome-ignore-start lint/a11y/useSemanticElements: the region role is what names the scrollable block for AT
    <pre
      tabIndex={0}
      role="region"
      aria-label={label ? `${label} code` : "Code"}
      className={`border-code-border bg-code-surface text-mono-micro text-code-foreground overflow-x-auto border px-4 py-3.5 leading-[1.0625rem] whitespace-pre-wrap ${
        arrangement === "labeled" ? "border-t-0" : ""
      } ${isBounded ? "h-56 overflow-y-auto" : ""} ${
        arrangement === "bare" && canCopy ? "pr-14" : ""
      }`}
    >
      {children ?? value}
    </pre>
    // biome-ignore-end lint/a11y/noNoninteractiveTabindex: see above
    // biome-ignore-end lint/a11y/useSemanticElements: see above
  )

  if (arrangement === "bare") {
    return (
      <div className={`relative ${className ?? ""}`}>
        {block}
        {canCopy ? (
          // On the code's own control fill rather than the page's, because it
          // sits on the dark surface. 32px square, which is under the touch
          // floor, so the target is bled outward with a pseudo-element.
          <button
            type="button"
            onClick={copy}
            aria-label={copyLabel}
            className="bg-code-control text-code-foreground absolute top-2 right-2 flex size-8 items-center justify-center before:absolute before:-inset-1.5 before:content-['']"
          >
            {copied ? (
              <FiCheck aria-hidden className="size-3.5" />
            ) : (
              <FiCopy aria-hidden className="size-3.5" />
            )}
          </button>
        ) : null}
      </div>
    )
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
            // Named for the block it copies, because these come in runs: the
            // bundled guide puts twenty on one page and the setup sheet puts
            // one beside the key field's own. "Copy" twenty times over is a
            // list a screen reader cannot choose from. The visible word stays
            // inside the name, which is what keeps voice control working.
            aria-label={copied ? undefined : copyLabel}
            className="relative opacity-75 before:absolute before:-inset-x-2 before:-inset-y-[0.6875rem] before:content-[''] hover:opacity-100"
          >
            {copied ? "Copied" : "Copy"}
          </button>
        ) : null}
      </div>
      {/* `border-t-0` on the block, because the label row already drew that edge. */}
      {block}
    </div>
  )
}
