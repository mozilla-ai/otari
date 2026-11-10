const SECOND = 1_000
const MINUTE = 60 * SECOND
const HOUR = 60 * MINUTE
const DAY = 24 * HOUR

// The steps a person reads off a time axis: 1, 2 and 5 below a second, then
// the divisions of a minute, an hour and a day rather than of a thousand.
const STEPS = [
  1,
  2,
  5,
  10,
  20,
  50,
  100,
  200,
  500,
  SECOND,
  2 * SECOND,
  5 * SECOND,
  10 * SECOND,
  15 * SECOND,
  30 * SECOND,
  MINUTE,
  2 * MINUTE,
  5 * MINUTE,
  10 * MINUTE,
  15 * MINUTE,
  30 * MINUTE,
  HOUR,
  2 * HOUR,
  3 * HOUR,
  6 * HOUR,
  12 * HOUR,
  DAY,
]

/** The smallest round step that keeps the axis ticks at least `minTickPx` apart. */
export function tickStep(
  totalMs: number,
  lanePx: number,
  minTickPx: number,
): number {
  const count = Math.max(1, Math.floor(lanePx / minTickPx))
  const raw = totalMs / count
  const step = STEPS.find((candidate) => candidate >= raw)
  if (step !== undefined) return step
  // Past a day: 1, 2 or 5 times a power of ten, in days.
  const days = raw / DAY
  const pow = 10 ** Math.floor(Math.log10(days))
  return ([1, 2, 5].find((m) => m * pow >= days) ?? 10) * pow * DAY
}
