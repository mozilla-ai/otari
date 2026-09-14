import { Tooltip as HeroTooltip } from "@heroui/react"
import type { JSX, ReactElement, ReactNode } from "react"

/** The props HeroUI's trigger hands a control: focus and hover handlers, a ref, and the `aria-describedby` that names the tooltip. Spread them onto the element, all of them. */
export type TooltipTriggerProps = JSX.IntrinsicElements["button"]

/**
 * How long a pointer has to rest on a trigger before the label appears.
 *
 * HeroUI reads `--tooltip-delay` off the document root and ships it at 1.5s,
 * which is react-aria's warmup default: long enough that a row of icon actions
 * reads as having no labels at all, because nobody holds a cursor still for a
 * second and a half to find out what a glyph does. A third of a second is short
 * enough to feel like an answer and long enough that sweeping the pointer
 * across a row of them does not flash four labels on the way past. Set as a
 * prop rather than in the theme so it travels with the component that depends
 * on it; react-aria still opens the next tooltip in a group instantly once one
 * has been seen.
 */
const OPEN_DELAY_MS = 300

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
    <HeroTooltip.Root delay={OPEN_DELAY_MS}>
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
