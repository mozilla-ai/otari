import { ProgressBar as HeroProgressBar } from "@heroui/react"

/**
 * A task on its way to finishing.
 *
 * Not `Meter`, and the difference is the question each answers. A meter is a
 * *measurement against a limit* that will sit there indefinitely: spend against
 * a budget, keys against a quota, and it is meaningful at rest. A progress bar
 * describes work in flight (a pricing refresh over 400 models, a re-encryption
 * pass) and is meaningless once it is done, because it disappears.
 *
 * Reaching for the wrong one is visible to an operator: a meter at 100% says
 * "you are at your limit", a progress bar at 100% says "finished".
 */
export function ProgressBar({
  value,
  max = 100,
  label,
  valueLabel,
  isIndeterminate,
  className = "",
}: {
  /** Ignored while `isIndeterminate`. */
  value?: number
  max?: number
  /**
   * What is being done ("Refreshing prices"). Required, because a bar has no
   * text of its own and motion-and-access.md's rule for `Meter` applies for the
   * same reason: a bar with no name is a decoration.
   */
  label: string
  /**
   * The progress in words ("120 of 400 models"). A count says more than a
   * percentage about work an operator is waiting on, which is why this is a
   * string rather than a formatter's output: only the caller knows the unit.
   */
  valueLabel?: string
  /**
   * For work whose total is unknown. It animates rather than filling, so it
   * must not be paired with a `valueLabel` claiming a count.
   */
  isIndeterminate?: boolean
  className?: string
}) {
  return (
    <HeroProgressBar
      aria-label={label}
      value={value}
      maxValue={max}
      isIndeterminate={isIndeterminate}
      className={`flex flex-col gap-1 ${className}`}
    >
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-caption">{label}</span>
        {valueLabel ? (
          // Mono, and tabular, because it is a count that ticks upward: a
          // proportional figure that changes width on every update makes the
          // line jitter, which is the same reason every counting value on this
          // dashboard is mono.
          <span className="font-mono text-mono-caption">{valueLabel}</span>
        ) : null}
      </div>
      <HeroProgressBar.Track>
        <HeroProgressBar.Fill />
      </HeroProgressBar.Track>
    </HeroProgressBar>
  )
}
