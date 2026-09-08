import type { ReactNode } from "react"

/**
 * "There is nothing here", in one treatment.
 *
 * 14px muted, which is the table's, and it is the table's because that is the
 * one the user picked when two of these turned up on the same screen at
 * different sizes: a chart band's message at 12px above a table's at 14px, in a
 * product where the two mean the same thing. Shared so the size cannot drift
 * again, the same reason `SettingsGroup` is shared.
 *
 * `minHeightClass` because an empty chart band still has to hold the space its
 * chart would have taken, or the page reflows the moment data arrives. It
 * REPLACES the vertical padding rather than joining it, which is why it is its
 * own prop and not a `className`: two padding utilities in one class list are
 * both plain class selectors, so the winner is whichever Tailwind emits last
 * (`py-10`, measured in the built stylesheet) and not whichever the call site
 * writes last. The inline `style` this replaces won that fight by being inline.
 */
export function EmptyMessage({
  children,
  minHeightClass,
}: {
  children: ReactNode
  /**
   * A min-height utility for a band that must not collapse, in `rem`. The
   * height is the space, so the padding comes off with it.
   */
  minHeightClass?: string
}) {
  return (
    <div
      className={`flex items-center justify-center px-4 text-center text-sm text-muted ${
        minHeightClass ?? "py-10"
      }`}
    >
      {children}
    </div>
  )
}
