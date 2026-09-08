import type { ReactNode, RefObject } from "react"

/**
 * A row's action: 13px of muted text, no border, no fill, no box.
 *
 * A row has three or four of these and boxing each one turned the last lane
 * into a control panel, louder than the row it acts on.
 *
 * `isDanger` is for the ARMED state and nothing else, which is worth spelling
 * out because it was got wrong at six sites on the first pass: a destructive
 * action at rest is the same muted text as its neighbors, and the ink arrives
 * only once the next click commits something. A row that paints Remove red
 * before anybody has touched it spends the color on a state nothing is in, and
 * by the time it means something the reader has stopped seeing it. Same rule
 * the spend figure follows: the color marks what you are about to do, never
 * the control that offers it.
 *
 * An action whose confirmation is a dialog rather than an inline arm stays
 * muted throughout: the dialog is where the danger lives.
 *
 * Shared once two pages had spelled it. Pair with `RowActionRow`, which sets the
 * 16px between them.
 */
export function RowAction({
  onPress,
  isDanger,
  isDisabled,
  ariaLabel,
  ref,
  children,
}: {
  onPress: () => void
  isDanger?: boolean
  isDisabled?: boolean
  /**
   * Forwarded to the button so a caller can move focus onto it. The two-step
   * confirm is the only caller: its swap unmounts a focused control, and a ref
   * is the only way to hand the caret to whatever replaced it.
   */
  ref?: RefObject<HTMLButtonElement | null>
  /**
   * Replaces the visible label for assistive tech, which is how a row action
   * says which row it acts on and why it is refused: a disabled control takes
   * no focus, so a tooltip reaches a pointer and nothing else, and the reason
   * has to be in the name.
   */
  ariaLabel?: string
  children: ReactNode
}) {
  return (
    <button
      ref={ref}
      type="button"
      disabled={isDisabled}
      aria-label={ariaLabel}
      onClick={onPress}
      className={`text-caption whitespace-nowrap transition-colors motion-reduce:transition-none disabled:opacity-(--disabled-opacity) ${
        isDanger ? "text-danger" : "hover:text-foreground"
      }`}
    >
      {children}
    </button>
  )
}

/** The lane those sit in: right-aligned, 16px apart. */
export function RowActionRow({ children }: { children: ReactNode }) {
  return <div className="flex items-center justify-end gap-4">{children}</div>
}
