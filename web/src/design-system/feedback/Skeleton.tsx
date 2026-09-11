/**
 * A placeholder holding the space a value will take.
 *
 * The rule for reaching for one is in layout-stability.md and is not "show
 * something while loading": a skeleton is right when the shape of what is
 * coming is *known* (a table of rows, a strip of four KPIs), and wrong when it
 * is not, because a skeleton that guesses wrong moves the page twice instead of
 * once. Where the shape is unknown, `PageLoading` is the honest answer.
 *
 * Sized by the caller, in `rem`, because only the caller knows what is coming.
 * The default is one line of body text, which is the most common case.
 */
export function Skeleton({
  className = "h-5 w-full",
  ariaLabel,
}: {
  /** The box to hold. A width and a height, nothing else. */
  className?: string
  /**
   * A name, for the rare skeleton that is the only thing on screen. Left off,
   * the box is hidden from assistive tech, which is right when a region around
   * it already says it is loading: a screen reader announcing eight
   * "loading" boxes for one table is worse than silence.
   */
  ariaLabel?: string
}) {
  return (
    <span
      {...(ariaLabel
        ? { role: "status", "aria-label": ariaLabel }
        : { "aria-hidden": true })}
      // `animate-pulse` guarded, like every other transition in the product:
      // an animation that runs until data arrives is exactly the kind a
      // vestibular disorder cannot dismiss.
      className={`block animate-pulse bg-surface-subtle motion-reduce:animate-none ${className}`}
    />
  )
}
