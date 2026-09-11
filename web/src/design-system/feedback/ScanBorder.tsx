import type { ReactNode } from "react"

/** Which ink the traveling arc is drawn in. */
export type ScanTone = "accent" | "danger"

const TONE_INK: Record<ScanTone, string> = {
  accent: "[--scan-ink:var(--color-primary)]",
  danger: "[--scan-ink:var(--color-danger)]",
}

const TONE_EDGE: Record<ScanTone, string> = {
  accent: "border-border",
  danger: "border-danger",
}

/**
 * A band whose edge is swept by a traveling arc while something is awaited.
 *
 * The one piece of decorative motion in this system, and it earns its place by
 * being literally true: it runs only while the product is watching for
 * something that has not arrived, and stops when it has. Anywhere else, motion
 * on an edge is noise.
 *
 * The arc is painted by the `otari-scan-border` rule in `globals.css` (a
 * masked conic gradient on an `::after`, with its angle animated through an
 * `@property`), so this component owns the geometry and the tone and nothing
 * else. The resting hairline is a real border underneath, which is what the
 * band shows when `isActive` is false and what the arc travels over when it is
 * true.
 *
 * `tone` picks the arc's ink through a variable rather than a second rule, so
 * a caller reporting a failure turns the sweep red without the stylesheet
 * having to know what a failure is.
 *
 * Under `prefers-reduced-motion` the arc holds still instead of vanishing: the
 * band should still read as the thing on the page that is waiting.
 */
export function ScanBorder({
  isActive,
  tone = "accent",
  className,
  children,
}: {
  /** Whether the arc travels. A stalled or finished wait sets this false. */
  isActive: boolean
  tone?: ScanTone
  className?: string
  children: ReactNode
}) {
  return (
    <div
      className={`relative border ${TONE_EDGE[tone]} ${
        isActive ? `otari-scan-border ${TONE_INK[tone]}` : ""
      } ${className ?? ""}`}
    >
      {children}
    </div>
  )
}
