import { useState } from "react"
import { FiChevronRight } from "react-icons/fi"

import { Markdown } from "@/design-system/content/Markdown"

/**
 * A reply's chain-of-thought, collapsed.
 *
 * Collapsed by default and not hidden: reasoning is usually longer than the
 * answer, so showing it inline buries what somebody asked for, while dropping it
 * loses the one thing that explains a surprising reply. A `<details>`-shaped
 * disclosure rather than the shared `Disclosure` primitive because this sits
 * inside a message bubble and carries no heading, border or padding of its own,
 * which is most of what that component is.
 *
 * It renders while the reasoning is still streaming, so the label is present
 * from the first fragment: a control that appears halfway through a reply moves
 * the answer under the reader's eye.
 */
export function ThinkingBlock({ content }: { content: string }) {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <div className="mb-2 flex flex-col gap-1.5">
      <button
        type="button"
        className="flex w-fit items-center gap-1.5 text-caption transition-colors hover:text-foreground"
        onClick={() => setIsOpen(!isOpen)}
        aria-expanded={isOpen}
      >
        <FiChevronRight
          aria-hidden
          className={`size-3 transition-transform ${isOpen ? "rotate-90" : ""}`}
        />
        Thinking
      </button>
      {isOpen ? (
        <div className="border-border border-l-2 pl-3 text-muted">
          <Markdown className="text-sm">{content}</Markdown>
        </div>
      ) : null}
    </div>
  )
}
