import { Tooltip as HeroTooltip } from "@heroui/react"
import type { ReactNode } from "react"

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
 * It wraps the trigger rather than taking it as a prop, so the trigger keeps
 * its own type: an `IconButton` inside one is still an `IconButton`, with its
 * required label and its 44px box intact. Passing it as `trigger={<Button/>}`
 * would have made the tooltip the parent of a node it cannot type-check.
 */
export function Tooltip({
  content,
  placement = "top",
  children,
}: {
  /** The label. A phrase, not a sentence, and never the only place it is said. */
  content: ReactNode
  placement?: "top" | "bottom" | "left" | "right"
  children: ReactNode
}) {
  return (
    <HeroTooltip.Root>
      {/* `inline-flex` rather than the element override HeroUI offers: a
          tooltip most often wraps a button sitting in a row of them, and the
          trigger's default block display would break the row's flex layout.
          v3 overrides an element through a `render` prop rather than `as`, and
          routing a div's prop type onto a span to gain nothing over a display
          utility is not worth the cast. */}
      <HeroTooltip.Trigger className="inline-flex">
        {children}
      </HeroTooltip.Trigger>
      <HeroTooltip.Content placement={placement}>{content}</HeroTooltip.Content>
    </HeroTooltip.Root>
  )
}
