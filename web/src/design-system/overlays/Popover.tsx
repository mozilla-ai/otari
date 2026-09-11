import { Popover as HeroPopover } from "@heroui/react"
import type { ReactNode } from "react"

/**
 * A panel anchored to the control that opened it.
 *
 * The line against `Tooltip`: a tooltip labels, a popover holds content an
 * operator interacts with, so this one takes focus and is dismissed
 * deliberately. The line against `Dialog`: a dialog is modal because it wants
 * the whole screen's attention, a popover stays anchored because what it says
 * is about the thing it points at.
 *
 * Uncontrolled by default, which is the opposite of `Dialog` and deliberate: a
 * popover is opened by its own trigger, which is inside it, so it can own that
 * state. Pass `isOpen` and `onOpenChange` for the case where something else
 * closes it (a route change, a mutation landing).
 */
export function Popover({
  trigger,
  children,
  placement = "bottom",
  isOpen,
  onOpenChange,
}: {
  /**
   * The control that opens it. Rendered as-is, so it keeps its own accessible
   * name and its own hit area.
   */
  trigger: ReactNode
  children: ReactNode
  placement?: "top" | "bottom" | "left" | "right"
  isOpen?: boolean
  onOpenChange?: (isOpen: boolean) => void
}) {
  return (
    <HeroPopover.Root isOpen={isOpen} onOpenChange={onOpenChange}>
      <HeroPopover.Trigger className="inline-flex">
        {trigger}
      </HeroPopover.Trigger>
      <HeroPopover.Content placement={placement}>
        {/* `Dialog` here is HeroUI's popover dialog, not our modal of the same
            name: it is what puts the panel in the accessibility tree as a
            dialog and traps focus inside it while it is open. Without it the
            panel is a div that a keyboard operator tabs straight past. */}
        <HeroPopover.Dialog>{children}</HeroPopover.Dialog>
      </HeroPopover.Content>
    </HeroPopover.Root>
  )
}
