import { Disclosure as HeroDisclosure } from "@heroui/react"
import type { ReactNode } from "react"

/**
 * A heading that opens to reveal what is under it.
 *
 * Right for detail an operator asks for (a request's raw payload, a provider's
 * full error), and wrong for anything they need in order to decide: content
 * behind a click is content most people never read, so a collapsed section is a
 * decision that it is optional.
 *
 * The chevron travels on the 150ms rung and is guarded, like the rail rows it
 * shares that duration with. HeroUI's own indicator carries the rotation, which
 * is why there is no transform spelled here.
 */
export function Disclosure({
  heading,
  children,
  isDefaultExpanded = false,
  isFlush = false,
}: {
  /** The always-visible label. It is the button, so it says what opening reveals. */
  heading: ReactNode
  children: ReactNode
  isDefaultExpanded?: boolean
  /**
   * Starts the content at the heading's left edge instead of HeroUI's 8px
   * inset. For a disclosure that continues a form, whose fields above it set
   * the edge.
   */
  isFlush?: boolean
}) {
  return (
    <HeroDisclosure.Root defaultExpanded={isDefaultExpanded}>
      <HeroDisclosure.Heading>
        <HeroDisclosure.Trigger className="flex min-h-11 w-full items-center justify-between gap-2 text-body">
          {heading}
          <HeroDisclosure.Indicator />
        </HeroDisclosure.Trigger>
      </HeroDisclosure.Heading>
      <HeroDisclosure.Content
        className={isFlush ? "[&>[data-slot=disclosure-body]]:px-0" : undefined}
      >
        <HeroDisclosure.Body className="pb-3">{children}</HeroDisclosure.Body>
      </HeroDisclosure.Content>
    </HeroDisclosure.Root>
  )
}
