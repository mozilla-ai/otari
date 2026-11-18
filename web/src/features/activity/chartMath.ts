/** Pure rules of the strip chart, kept apart from the drawing so they can be tested. */

export interface Bucket {
  requests: number
  errors: number
}

const MANTISSAS = [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8]

/**
 * The y step: the smallest of {1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8} x 10^k that
 * reaches `max / (tickCount - 1)`, rounded up to an integer (these are counts).
 */
export function yStep(max: number, tickCount: number): number {
  const target = max / Math.max(1, tickCount - 1)
  if (!(target > 0)) return 1
  for (let k = -1; k < 15; k += 1) {
    for (const mantissa of MANTISSAS) {
      const candidate = mantissa * 10 ** k
      if (candidate >= target - 1e-9)
        return Math.max(1, Math.ceil(candidate - 1e-9))
    }
  }
  return Math.ceil(target)
}

/** Every tick from 0 to the top, `tickCount` of them, and the top of the scale. */
export function yTicks(
  max: number,
  tickCount: number,
): { ticks: number[]; top: number } {
  const step = yStep(max, tickCount)
  const ticks = Array.from({ length: tickCount }, (_, index) => index * step)
  return { ticks, top: step * (tickCount - 1) }
}

/** Room between two x labels. */
const LABEL_GAP = 12

/** A generous width for the widest x label at the axis's 12px. */
export function xLabelWidth(labels: readonly string[]): number {
  return labels.reduce((widest, label) => Math.max(widest, label.length), 0) * 7
}

/**
 * Every k-th bucket counted back from the latest, so the latest is always
 * labelled, spaced so labels `labelWidth` wide never touch. The latest label
 * ends at the plot's right edge rather than centering on its bucket, so the one
 * before it keeps half a label further away; the earliest is dropped when it
 * would run past `leftRoom` to the left of the plot.
 */
export function xTickIndices(
  count: number,
  pitch: number,
  minGap: number,
  labelWidth = 0,
  leftRoom = Number.POSITIVE_INFINITY,
): number[] {
  if (count <= 0) return []
  const step = Math.max(pitch, 0.0001)
  const k = Math.max(
    1,
    Math.ceil(Math.max(minGap, labelWidth + LABEL_GAP) / step),
  )
  const first = Math.max(
    k,
    Math.ceil((1.5 * labelWidth + LABEL_GAP) / step - 0.5),
  )
  const out: number[] = []
  for (
    let index = count - 1;
    index >= 0;
    index -= index === count - 1 ? first : k
  )
    out.push(index)
  const earliest = out[out.length - 1]
  if (out.length > 1 && (earliest + 0.5) * step - labelWidth / 2 < -leftRoom)
    out.pop()
  return out.reverse()
}

/** Bars touch below this many pixels per bucket. */
const TOUCHING_PITCH = 4

// Recharts takes `barCategoryGap` as the space on EACH side of a bar, so the
// "25% gap between bars" the design asks for (bar 26px, gap 9px at 31 buckets)
// is 12.5% here. A bare "25%" draws bars half the slot wide (18px).
const GAP_EACH_SIDE = 0.125

export function barCategoryGap(pitch: number): `${number}%` | 0 {
  return pitch < TOUCHING_PITCH ? 0 : `${GAP_EACH_SIDE * 100}%`
}

/**
 * Where a bar sits inside its slot, by the rule Recharts applies (a gap each
 * side, the size rounded to whole pixels, capped at `maxBarSize`), so marks
 * drawn beside the bars have the bars' own width and left edge.
 */
export function barBox(
  pitch: number,
  gap: `${number}%` | 0,
  maxBarSize = 40,
): { offset: number; size: number } {
  const side = gap === 0 ? 0 : pitch * GAP_EACH_SIDE
  const raw = pitch - 2 * side
  const natural = raw > 1 ? Math.round(raw) : raw
  const size = Math.min(natural, maxBarSize)
  return { offset: side + (natural - size) / 2, size }
}

/** The readout's word for a count of its unit: "1 request", "2 requests". */
export function unitFor(count: number, unit: string): string {
  return count === 1 ? unit.replace(/s$/, "") : unit
}

export function minPointSize(value: number | null | undefined): number {
  return value != null && value > 0 ? 2 : 0
}

/** The sum of both series over an inclusive bucket range. */
export function sumRange(
  series: Bucket[],
  startIndex: number,
  endIndex: number,
): { count: number; failed: number } {
  let count = 0
  let failed = 0
  const lo = Math.max(0, Math.min(startIndex, endIndex))
  const hi = Math.min(series.length - 1, Math.max(startIndex, endIndex))
  for (let index = lo; index <= hi; index += 1) {
    count += series[index].requests
    failed += Math.min(series[index].errors, series[index].requests)
  }
  return { count, failed }
}

/** The tallest bucket of an inclusive range, for a series whose buckets overlap and cannot be summed. */
export function peakRange(
  series: Bucket[],
  startIndex: number,
  endIndex: number,
): { count: number; failed: number } {
  let count = 0
  let failed = 0
  const lo = Math.max(0, Math.min(startIndex, endIndex))
  const hi = Math.min(series.length - 1, Math.max(startIndex, endIndex))
  for (let index = lo; index <= hi; index += 1) {
    count = Math.max(count, series[index].requests)
    failed = Math.max(
      failed,
      Math.min(series[index].errors, series[index].requests),
    )
  }
  return { count, failed }
}

export interface Reading {
  count: number
  // Null: this chart has no failed figure to show.
  failed: number | null
  // A tallest-bucket reading rather than a sum.
  peak?: boolean
}

/**
 * What the header reads with nothing under the pointer. Totals the caller
 * gives win (null: still loading). Otherwise the window's buckets are summed,
 * unless buckets overlap and cannot be summed, when the tallest one stands in.
 * Returns null while loading.
 */
export function windowReading(input: {
  totals: { count: number; failed: number | null } | null | undefined
  additive: boolean
  loading: boolean
  series: Bucket[]
  startIndex: number
  endIndex: number
}): Reading | null {
  const { totals, additive, loading, series, startIndex, endIndex } = input
  if (
    totals === null ||
    (totals === undefined && loading && series.length === 0)
  ) {
    return null
  }
  if (totals !== undefined)
    return { count: totals.count, failed: totals.failed }
  return additive
    ? sumRange(series, startIndex, endIndex)
    : { ...peakRange(series, startIndex, endIndex), peak: true }
}

/**
 * What the header reads over a bucket or a dragged range. One bucket is its
 * own value; a range is summed, or its tallest bucket when the series cannot
 * be summed. A chart with no failed figure keeps having none.
 */
export function pointedReading(input: {
  base: Reading | null
  additive: boolean
  series: Bucket[]
  lo: number
  hi: number
}): Reading | null {
  const { base, additive, series, lo, hi } = input
  if (!base) return null
  const single = lo === hi
  const values =
    single || additive ? sumRange(series, lo, hi) : peakRange(series, lo, hi)
  return {
    ...values,
    peak: !additive && !single,
    ...(base.failed === null ? { failed: null } : {}),
  }
}
