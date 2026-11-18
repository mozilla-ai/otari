const DAY_MS = 86_400_000

/**
 * A limit's pace over its period so far: what it lands on at the period's end,
 * and when it runs out if that is over the limit. Nothing for a period shorter
 * than a week or less than a tenth gone, or with nothing spent: a projection
 * there says less than the meter under it.
 */
export function paceOf({
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
}):
  | { projected: number; runsOutAt: number | undefined; end: number }
  | undefined {
  if (!periodStart || !periodEnd || spent <= 0) return undefined
  const start = Date.parse(periodStart)
  const end = Date.parse(periodEnd)
  const length = end - start
  const gone = (now - start) / length
  if (length < 7 * DAY_MS || gone < 0.1 || gone >= 1) return undefined
  const projected = spent / gone
  return {
    projected,
    runsOutAt:
      projected > max
        ? now + (max - spent) / (spent / (now - start))
        : undefined,
    end,
  }
}
