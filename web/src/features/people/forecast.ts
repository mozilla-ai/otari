const HOUR_MS = 3_600_000

/**
 * When a spend limit runs out at the pace spent since its period began: a
 * time, or undefined when it lasts the period or is $0, which has no pace to
 * run out at. The first six hours of a period count as six, so one early
 * request does not read as a day's pace.
 */
export function runsOutAt({
  spent,
  max,
  periodStart,
  periodEnd,
  now,
}: {
  spent: number
  max: number
  periodStart: string | null | undefined
  periodEnd: string | null | undefined
  now: number
}): number | undefined {
  if (max <= 0) return undefined
  if (spent >= max) return now
  const start = periodStart ? Date.parse(periodStart) : Number.NaN
  if (!Number.isFinite(start) || spent <= 0) return undefined
  const elapsed = Math.max(now - start, 6 * HOUR_MS)
  const at = now + (max - spent) / (spent / elapsed)
  const end = periodEnd ? Date.parse(periodEnd) : Number.NaN
  return Number.isFinite(end) && at >= end ? undefined : at
}
