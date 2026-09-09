import type { ReactNode } from "react"

import { Button, type ButtonProps } from "./Button"

/**
 * A button whose whole label is a glyph.
 *
 * Two things it enforces that a `Button` with an icon in it cannot. The
 * accessible name is a **required** prop rather than an optional `aria-label`,
 * because a control with no text and no name is unreachable by speech input and
 * anonymous to a screen reader, and "someone will remember the aria-label" is
 * how that ships. And the 44px touch floor is on the box rather than in
 * padding, so the target does not depend on the glyph's own size.
 *
 * The glyph is sized by the caller (`size-4`, or `h-3.5 w-3.5` in a dense row)
 * and is always `aria-hidden`: it is decoration once `label` carries the name.
 */
export function IconButton({
  label,
  children,
  className = "",
  ...rest
}: Omit<ButtonProps, "children" | "aria-label"> & {
  /** The accessible name. Says what the control does, not what the glyph is. */
  label: string
  children: ReactNode
}) {
  return (
    <Button
      aria-label={label}
      // A square target at the touch floor. `min-h-11 min-w-11` rather than a
      // height: a glyph that grows (a two-character count, a wider chevron)
      // grows the box instead of overflowing it.
      className={`min-h-11 min-w-11 justify-center ${className}`}
      {...rest}
    >
      {children}
    </Button>
  )
}
