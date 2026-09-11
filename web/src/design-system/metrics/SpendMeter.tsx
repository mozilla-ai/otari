/** How far along its allocation a spend is, which decides how it is drawn. */
export type SpendState = "on-track" | "near-limit" | "over"

/**
 * The one place a spend is classified, so the bar and the figure beside it
 * cannot disagree about which of the three states this is.
 *
 * `nearLimitAt` is the share of the allocation past which spend stops being
 * unremarkable. It is a fraction of the limit and not an absolute, because
 * "nearly out" means the same thing on a $50 budget and a $5,000 one.
 */
export function spendState(
  spent: number,
  allocated: number,
  nearLimitAt = 0.8,
): SpendState {
  // A cap of zero admits nothing, so any spend against one is past it. Answered
  // from the spend rather than by dividing, which is what keeps a zero
  // allocation from producing Infinity or NaN here.
  if (allocated <= 0) return spent > 0 ? "over" : "on-track"
  if (spent > allocated) return "over"
  return spent >= allocated * nearLimitAt ? "near-limit" : "on-track"
}

/**
 * A spend against its allocation, in three states rather than the two the code
 * carried before, where everything under the limit looked identical and the
 * first thing anyone learned was that they had already gone past.
 *
 * | state      | bar                                    | figure  |
 * | ---------- | -------------------------------------- | ------- |
 * | on-track   | accent to the spend                    | normal  |
 * | near-limit | accent to the threshold, danger beyond | normal  |
 * | over       | danger, full width                     | danger  |
 *
 * The middle state is two segments and not a second color for the whole bar,
 * which is what makes it legible without hue: the overshoot past the threshold
 * is a distinct block with a hairline of track showing between it and the
 * accent, so in grayscale, or to anyone who cannot separate teal from red, the
 * bar still says "there is a part of this that is past the mark". A single
 * recolored bar would say nothing at all under those conditions.
 *
 * The figure changing color is reserved for `over`, and it is the only number
 * anywhere in this product that changes color. That is the point: it has to be
 * worth something when it happens.
 */
export function SpendMeter({
  spent,
  allocated,
  ariaLabel,
  nearLimitAt = 0.8,
  className = "",
}: {
  spent: number
  allocated: number
  ariaLabel: string
  nearLimitAt?: number
  className?: string
}) {
  const state = spendState(spent, allocated, nearLimitAt)
  // Full for spend against a zero cap, matching the state above: a bar drawn
  // empty beside a reading of "over budget" contradicts itself.
  const share = allocated > 0 ? spent / allocated : spent > 0 ? 1 : 0
  const pct = Math.max(0, Math.min(1, share)) * 100
  const thresholdPct = nearLimitAt * 100
  return (
    <span
      role="progressbar"
      aria-label={ariaLabel}
      // Clamped, because a progressbar's value is documented to sit inside its
      // range and a widget that reports 137 out of 100 is malformed. The number
      // that matters is not lost: it goes in `aria-valuetext`, which is the
      // field for the human reading of a value, so a screen reader is told both
      // that the bar is full and that the spend is 37% past the limit. Neither
      // field is abused to carry the other's fact.
      aria-valuenow={Math.min(100, Math.round(share * 100))}
      aria-valuetext={
        state === "over"
          ? `${Math.round(share * 100)}% of limit — over budget`
          : `${Math.round(share * 100)}% of limit`
      }
      aria-valuemin={0}
      aria-valuemax={100}
      className={`flex h-[0.1875rem] w-full bg-surface-subtle ${className}`}
    >
      {state === "over" ? (
        <span className="block h-full w-full bg-danger" />
      ) : state === "near-limit" ? (
        <>
          <span
            className="block h-full bg-accent"
            style={{ width: `${thresholdPct}%` }}
          />
          {/* The hairline that makes the two segments read as two. It is track,
              not a border, so it cannot pick up a color of its own. */}
          <span className="block h-full w-px shrink-0" />
          <span
            className="block h-full bg-danger"
            style={{ width: `${Math.max(pct - thresholdPct, 0)}%` }}
          />
        </>
      ) : (
        <span className="block h-full bg-accent" style={{ width: `${pct}%` }} />
      )}
    </span>
  )
}
