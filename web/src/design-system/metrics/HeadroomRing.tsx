import { spendState } from "./SpendMeter"

/**
 * What is left of a limit, as a ring that starts full and empties clockwise,
 * with the figure beside it.
 *
 * `used` is a **fraction, not a percentage**: `0.88` draws 12% of the ring and
 * prints "12% left". It may pass 1, which is over the limit.
 *
 * The state comes from `spendState`, so a ring and a `SpendMeter` drawn from the
 * same spend agree on which of the three states it is:
 *
 * | state      | ring                     | figure            |
 * | ---------- | ------------------------ | ----------------- |
 * | on-track   | accent arc on the track  | "N% left"         |
 * | near-limit | danger arc on the track  | "N% left"         |
 * | over       | danger track, no arc     | "Over limit", danger |
 *
 * The figure is always printed, so the ring is never the only channel: it is
 * hidden from assistive technology and the words carry the reading. The left
 * share rounds down, the tighter direction, so a limit with a cent spent never
 * reads "100% left" and one a cent from the cap reads "0% left", not "1%".
 */
export function HeadroomRing({
  used,
  nearLimitAt = 0.8,
}: {
  used: number
  nearLimitAt?: number
}) {
  const state = spendState(used, 1, nearLimitAt)
  // Rounded to a millionth before the ceiling, or float noise reads 90% used as
  // "9% left"; any use at all is at least 1%, however large the limit.
  const usedPct =
    used > 0 ? Math.max(1, Math.ceil(Math.round(used * 1e6) / 1e4)) : 0
  const leftPct = Math.max(0, Math.min(100, 100 - usedPct))
  return (
    <span className="inline-flex items-center gap-2 whitespace-nowrap tabular-nums">
      <svg
        viewBox="0 0 16 16"
        aria-hidden="true"
        className="size-3.5 shrink-0 -rotate-90"
      >
        <circle
          cx="8"
          cy="8"
          r="6"
          fill="none"
          strokeWidth="2.5"
          className={
            state === "over" ? "stroke-danger" : "stroke-surface-subtle"
          }
        />
        {state === "over" || leftPct === 0 ? null : (
          <circle
            cx="8"
            cy="8"
            r="6"
            fill="none"
            strokeWidth="2.5"
            // pathLength makes the dash a percentage of the circumference.
            pathLength={100}
            strokeDasharray={`${leftPct} 100`}
            className={
              state === "near-limit" ? "stroke-danger" : "stroke-accent"
            }
          />
        )}
      </svg>
      {state === "over" ? (
        <span className="text-danger">Over limit</span>
      ) : (
        `${leftPct}% left`
      )}
    </span>
  )
}
