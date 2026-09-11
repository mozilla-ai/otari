import { Tooltip as HeroTooltip } from "@heroui/react"
import type { JSX, ReactElement, ReactNode } from "react"

/** The props HeroUI's trigger hands a control: focus and hover handlers, a ref, and the `aria-describedby` that names the tooltip. Spread them onto the element, all of them. */
export type TooltipTriggerProps = JSX.IntrinsicElements["button"]

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
 * A HeroUI `Button` (so `IconButton`) cannot use the function form and keeps the
 * wrapper: its props are react-aria's rather than the DOM's, so the trigger's
 * `onClick` and `onPointerEnter` do not even typecheck against it. That is why
 * the props are typed for a `button` here rather than generically.
 *
 * Either way the trigger keeps its own type, which is what a `trigger={<Button/>}`
 * prop could not have done: the tooltip would have been the parent of a node it
 * cannot type-check.
 */
export function Tooltip({
  content,
  placement = "top",
  children,
}: {
  /** The label. A phrase, not a sentence, and never the only place it is said. */
  content: ReactNode
  placement?: "top" | "bottom" | "left" | "right"
  children: ReactNode | ((props: TooltipTriggerProps) => ReactElement)
}) {
  return (
    <HeroTooltip.Root>
      {typeof children === "function" ? (
        // The generic is what types the render function for a `button` rather
        // than for the `div` HeroUI's trigger defaults to. A control wanting a
        // tooltip is a button here; a link would need its own spelling.
        <HeroTooltip.Trigger<"button"> render={children} />
      ) : (
        // `inline-flex` rather than the element override HeroUI offers: a
        // tooltip most often wraps a button sitting in a row of them, and the
        // trigger's default block display would break the row's flex layout.
        <HeroTooltip.Trigger className="inline-flex">
          {children}
        </HeroTooltip.Trigger>
      )}
      <HeroTooltip.Content placement={placement}>{content}</HeroTooltip.Content>
    </HeroTooltip.Root>
  )
}
