import { Popover as HeroPopover } from "@heroui/react"
import type { ReactElement, ReactNode } from "react"

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
 *
 * The trigger is rendered as-is rather than wrapped in `HeroPopover.Trigger`,
 * which is what keeps one button in the accessibility tree instead of two (the
 * problem `design/overlays.md` describes for Tooltip). The cost is a contract
 * the type cannot state on its own: it has to be a react-aria pressable, which
 * in this codebase means our `Button` or `IconButton`. A `<span>` or a native
 * `<button>` renders and never opens the panel.
 */
export function Popover({
  trigger,
  children,
  label,
  placement = "bottom",
  padding = "default",
  isOpen,
  onOpenChange,
}: {
  /**
   * The control that opens it, rendered as-is so it keeps its own accessible
   * name and hit area. Must be a react-aria pressable: our `Button` or
   * `IconButton`, never a `<span>` or a native `<button>`.
   */
  trigger: ReactElement
  children: ReactNode
  /**
   * The panel's accessible name. A `<h2>` inside the panel does not supply one,
   * so without this the dialog is announced unnamed.
   */
  label: string
  placement?: "top" | "bottom" | "bottom end" | "left" | "right"
  /** Use none when child sections own their padding and full-width dividers. */
  padding?: "default" | "none"
  isOpen?: boolean
  onOpenChange?: (isOpen: boolean) => void
}) {
  return (
    <HeroPopover.Root isOpen={isOpen} onOpenChange={onOpenChange}>
      {trigger}
      <HeroPopover.Content placement={placement}>
        {/* `Dialog` here is HeroUI's popover dialog, not our modal of the same
            name: it is what puts the panel in the accessibility tree as a
            dialog and traps focus inside it while it is open. Without it the
            panel is a div that a keyboard operator tabs straight past. */}
        <HeroPopover.Dialog
          aria-label={label}
          className={padding === "none" ? "p-0" : undefined}
        >
          {children}
        </HeroPopover.Dialog>
      </HeroPopover.Content>
    </HeroPopover.Root>
  )
}
