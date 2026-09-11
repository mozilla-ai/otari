import type { ReactNode } from "react"
import { useEffect, useState } from "react"
import { Checkbox as AriaCheckbox } from "react-aria-components"

// The box visual, split out so it can hold optimistic state: react-aria only
// reports the new `isSelected` after the whole collection re-renders (O(rows)
// per click, tens to hundreds of ms on big pages or slow machines), which made
// the checkmark feel laggy. On pointerdown the visual flips immediately; the
// authoritative state catches up and clears the override, and a timeout clears
// it as a backstop if the press never lands (e.g. drag-away).
export function CheckboxVisual({
  isSelected,
  isIndeterminate,
  isDisabled,
}: {
  isSelected: boolean
  isIndeterminate: boolean
  isDisabled: boolean
}) {
  const [flash, setFlash] = useState<boolean | null>(null)

  useEffect(() => {
    if (flash !== null && isSelected === flash) setFlash(null)
  }, [isSelected, flash])
  useEffect(() => {
    if (flash === null) return
    const timer = setTimeout(() => setFlash(null), 600)
    return () => clearTimeout(timer)
  }, [flash])

  const showChecked = flash ?? (isSelected || isIndeterminate)
  return (
    <span
      onPointerDown={() => {
        if (!isDisabled) setFlash(!isSelected)
      }}
      // Square, and outlined in the control edge rather than the divider
      // border: `--color-border` is a 0.06 alpha tuned to separate two surfaces
      // of nearly the same value, which leaves a 16px box on the page ground
      // almost invisible. Checked drops the border entirely so the accent fill
      // is the whole shape.
      //
      // The glyph is white rather than `text-accent-foreground`, and that is
      // not the same call the filled buttons made. A checkmark is a non-text
      // graphic, held to 3:1, and white on the accent measures 3.48, which
      // clears it. A button label is text at 4.5, which is why the button
      // family had to darken its own ground to keep white ink and this box
      // does not have to. `--accent-foreground` stays where it is: it also
      // feeds HeroUI's own components on the accent, where the text floor
      // still applies, so the graphic gets its own token instead.
      className={`otari-checkbox-box flex h-4 w-4 items-center justify-center transition-colors ${
        showChecked
          ? "bg-control-indicator text-accent-glyph"
          : "border border-control-border bg-background"
      } group-data-[focus-visible]:otari-focus-ring`}
    >
      {isIndeterminate && flash === null ? (
        <svg
          viewBox="0 0 24 24"
          className="h-3 w-3"
          fill="none"
          stroke="currentColor"
          strokeWidth={3}
          aria-hidden="true"
        >
          <line x1="6" x2="18" y1="12" y2="12" strokeLinecap="round" />
        </svg>
      ) : showChecked ? (
        <svg
          viewBox="0 0 24 24"
          className="h-3 w-3"
          fill="none"
          stroke="currentColor"
          strokeWidth={3}
          aria-hidden="true"
        >
          <polyline
            points="5 12 10 17 19 7"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      ) : null}
    </span>
  )
}

/**
 * A checkbox on the design tokens.
 *
 * react-aria rather than HeroUI's own `Checkbox`, for the reason
 * `DataTable`'s selection box gives: HeroUI splits the control across
 * subcomponents and the two would not look alike. One visual serves both, so a
 * standalone checkbox and a table's selection box cannot drift apart.
 */
export function Checkbox({
  isSelected,
  onChange,
  isDisabled = false,
  ariaLabel,
  children,
}: {
  isSelected: boolean
  onChange: (isSelected: boolean) => void
  isDisabled?: boolean
  /**
   * An accessible name that says more than the visible label does, for a group
   * whose labels repeat across the page (one workspace list per guardrail, say).
   * Keep the visible text inside it, so speech input still reaches the control.
   */
  ariaLabel?: string
  children: ReactNode
}) {
  return (
    <AriaCheckbox
      aria-label={ariaLabel}
      isSelected={isSelected}
      onChange={onChange}
      isDisabled={isDisabled}
      className="group flex w-fit items-center gap-2 text-body"
    >
      {({ isSelected: selected, isDisabled: disabled }) => (
        <>
          <CheckboxVisual
            isSelected={selected}
            isIndeterminate={false}
            isDisabled={disabled}
          />
          {children}
        </>
      )}
    </AriaCheckbox>
  )
}
