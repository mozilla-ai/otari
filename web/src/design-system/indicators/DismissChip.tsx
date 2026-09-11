import type { ReactNode } from "react"
import { FiX } from "react-icons/fi"

/**
 * A value somebody chose, with the way to unchoose it.
 *
 * Identifier text with a remove control beside it, and no container: no border,
 * no fill, no shape of its own. Dismissability is a modifier on a kind rather
 * than a kind, so this wears the identifier kind and adds a control.
 *
 * No box, because the kinds here are told apart by dot, case and separator; a
 * bordered, filled chip would be the only boxed thing in the product.
 *
 * The dismiss is a 24px target holding a 12px glyph, which is under the 44px
 * floor `design/motion-and-access.md` sets, and knowingly so. The device that
 * doc names (a `before:` bleed, as `Toggle` uses) is not free here: this row
 * wraps at an 8px gap, so stacked bleeds would overlap by about 12px and a
 * press near the seam would dismiss the neighboring filter. Reaching 44px for
 * real instead grows the row on three pages. #947 carries the decision.
 */
export function DismissChip({
  label,
  value,
  onDismiss,
  dismissLabel,
}: {
  /** The dimension, shown before the value and divided from it by a colon. */
  label?: string
  value: ReactNode
  onDismiss: () => void
  /** Names the target for assistive tech; falls back to the label and value. */
  dismissLabel?: string
}) {
  return (
    <span className="inline-flex items-center gap-1.5 text-mono-caption text-foreground">
      {label ? <span className="text-subtle">{label}:</span> : null}
      {value}
      <button
        type="button"
        onClick={onDismiss}
        aria-label={
          dismissLabel ?? `Remove ${label ? `${label} ` : ""}${value}`
        }
        className="inline-flex h-6 w-6 items-center justify-center text-muted hover:text-foreground"
      >
        <FiX aria-hidden="true" className="h-3 w-3" />
      </button>
    </span>
  )
}
