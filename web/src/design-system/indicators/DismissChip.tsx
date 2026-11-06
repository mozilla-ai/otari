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
 * The dismiss is drawn as a 24px square holding a 12px glyph, and on a coarse
 * pointer its hit area grows to 44px through a `before:` bleed, as `Toggle`'s
 * does, so nothing moves. Two bleeds stacked in a wrapped row would overlap
 * unless the rows are 44px apart, so every row that holds these chips sets
 * `pointer-coarse:gap-y-5` (24px + 20px = 44px pitch). A new host does the same.
 * Under a fine pointer nothing changes.
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
        className="relative inline-flex h-6 w-6 items-center justify-center text-muted hover:text-foreground pointer-coarse:before:absolute pointer-coarse:before:-inset-2.5"
      >
        <FiX aria-hidden="true" className="h-3 w-3" />
      </button>
    </span>
  )
}
