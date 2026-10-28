import { Tooltip as HeroTooltip } from "@heroui/react"
import {
  isValidElement,
  type JSX,
  type ReactElement,
  type ReactNode,
} from "react"

import { Button } from "../actions/Button"
import { IconButton } from "../actions/IconButton"

/** The props HeroUI's trigger hands a control: focus and hover handlers, a ref, and the `aria-describedby` that names the tooltip. Spread them onto the element, all of them. */
export type TooltipTriggerProps = JSX.IntrinsicElements["button"]

/**
 * How long a pointer rests on a trigger before the label appears.
 *
 * Left unset, HeroUI reads `--tooltip-delay` off the document root and ships it
 * at react-aria's 1.5s warmup, which is long enough that a lane of icon actions
 * reads as having no labels at all. A third of a second answers without
 * flashing a label at every glyph a pointer crosses on the way past. A prop
 * rather than an override of that variable, so the behavior travels with the
 * component rather than with the app's stylesheet; react-aria still opens the
 * next tooltip in a group instantly once one has been seen.
 */
export const TOOLTIP_OPEN_DELAY_MS = 300

/**
 * A short label revealed by hovering or focusing the thing it describes.
 *
 * **A tooltip is never the only channel.** motion-and-access.md's rule about
 * hover applies hardest here: a tooltip does not exist on a phone, so anything
 * only it says is unsaid for a touch operator. That makes it right for a
 * *repetition* (spelling out an icon-only control that already has an
 * `aria-label`, giving an exact timestamp beside a relative one) and wrong for
 * anything an operator needs in order to act.
 *
 * Two shapes, and which one you want depends on whether the thing described is
 * itself a control:
 *
 * - **A node**, for content that is not interactive. It is wrapped in HeroUI's
 *   own trigger, which is a `div` the library gives `role="button"` and
 *   `tabIndex=0` so a keyboard can reach it at all.
 * - **A function**, for a native control. It receives the trigger's props and
 *   must spread every one of them onto its own element, the `ref` included. The
 *   wrapper's `role="button"` around a real button is the reason this form
 *   exists: it would announce "Edit button" inside "Edit button", take a tab
 *   stop of its own, and give every `getByRole("button")` two matches instead
 *   of one, which is a dozen assertions in this repo's e2e suite.
 *
 * A `Button` or `IconButton` cannot use the function form: its props are
 * react-aria's rather than the DOM's, so the trigger's `onClick` and
 * `onPointerEnter` do not even typecheck against it. It is a react-aria control,
 * though, so it reads the trigger state from the tooltip root directly, and it
 * is rendered with no wrapper at all: one button, and the label on it. That is
 * why the props are typed for a `button` here rather than generically.
 *
 * Either way the trigger keeps its own type, which is what a `trigger={<Button/>}`
 * prop could not have done: the tooltip would have been the parent of a node it
 * cannot type-check.
 */
export function Tooltip({
  content,
  placement = "top",
  delay = TOOLTIP_OPEN_DELAY_MS,
  isOpen,
  onOpenChange,
  children,
}: {
  /** The label. A phrase, not a sentence, and never the only place it is said. */
  content: ReactNode
  placement?: "top" | "bottom" | "left" | "right"
  /**
   * Milliseconds before it opens. `0` where hovering is how the value is read,
   * rather than a pointer passing a lane of icons on its way somewhere else.
   */
  delay?: number
  /** Held by the caller, where something besides hover and focus opens it, such as a tap. */
  isOpen?: boolean
  onOpenChange?: (isOpen: boolean) => void
  children: ReactNode | ((props: TooltipTriggerProps) => ReactElement)
}) {
  if (
    isValidElement(children) &&
    (children.type === Button || children.type === IconButton)
  ) {
    return (
      <HeroTooltip.Root
        delay={delay}
        isOpen={isOpen}
        onOpenChange={onOpenChange}
      >
        {children}
        <HeroTooltip.Content placement={placement}>
          {content}
        </HeroTooltip.Content>
      </HeroTooltip.Root>
    )
  }
  return (
    <HeroTooltip.Root delay={delay} isOpen={isOpen} onOpenChange={onOpenChange}>
      {typeof children === "function" ? (
        // The generic is what types the render function for a `button` rather
        // than for the `div` HeroUI's trigger defaults to. A control wanting a
        // tooltip is a button here; a link would need its own spelling.
        <HeroTooltip.Trigger<"button"> render={children} />
      ) : (
        // `inline-flex` rather than the element override HeroUI offers: a
        // tooltip most often wraps a button sitting in a row of them, and the
        // trigger's default block display would break the row's flex layout.
        // `cursor-default` because the wrapper's `role="button"` is for keyboard
        // reach, and the base layer's pointer for that role would make text look
        // clickable. A control inside keeps its own pointer.
        <HeroTooltip.Trigger className="inline-flex cursor-default">
          {children}
        </HeroTooltip.Trigger>
      )}
      <HeroTooltip.Content placement={placement}>{content}</HeroTooltip.Content>
    </HeroTooltip.Root>
  )
}
