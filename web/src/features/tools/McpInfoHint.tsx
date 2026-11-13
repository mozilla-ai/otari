import { type ReactNode, useState } from "react"
import { FiInfo } from "react-icons/fi"

import {
  Tooltip,
  type TooltipTriggerProps,
} from "@/design-system/overlays/Tooltip"

/**
 * An info icon whose tooltip explains the control beside it, in short
 * paragraphs. Hover and focus open it, and so does a press, so a touch
 * operator can read it too.
 *
 * A native button, as `RowAction` is: a react-aria `Button` placed inside a
 * `Select` takes the select's own trigger role, and this sits on a select's
 * label line.
 */
export function McpInfoHint({
  label,
  children,
}: {
  /** Names the icon for assistive technology, such as "Explain this switch". */
  label: string
  /** The explanation: one `<span>` per paragraph. */
  children: ReactNode
}) {
  const [isOpen, setOpen] = useState(false)
  return (
    <Tooltip
      content={
        <span className="flex max-w-xs flex-col gap-2 text-left">
          {children}
        </span>
      }
      isOpen={isOpen}
      onOpenChange={setOpen}
    >
      {({ className: _triggerClass, ...trigger }: TooltipTriggerProps) => (
        <button
          {...trigger}
          type="button"
          aria-label={label}
          onClick={() => setOpen((open) => !open)}
          className="relative flex size-6 shrink-0 items-center justify-center text-muted transition-colors before:absolute before:-inset-2.5 before:content-[''] hover:text-foreground motion-reduce:transition-none"
        >
          <FiInfo aria-hidden className="size-4" />
        </button>
      )}
    </Tooltip>
  )
}
