// The squarified treemap of Bruls, Huizing and van Wijk (2000): items sorted
// largest first are laid into rows along the shorter side of what is left, and a
// row grows while its worst aspect ratio improves. Pure, so it is tested for the
// two things a layout owes: each area follows its value, and the rectangles tile
// the bounds.

export interface Rect {
  x: number
  y: number
  width: number
  height: number
}

export type Placed<T> = Rect & { item: T }

// The worst aspect ratio (>= 1) of a row of areas laid along a side of `side`.
function worst(areas: readonly number[], side: number): number {
  let sum = 0
  let min = Number.POSITIVE_INFINITY
  let max = 0
  for (const area of areas) {
    sum += area
    min = Math.min(min, area)
    max = Math.max(max, area)
  }
  const square = side * side
  return Math.max((square * max) / (sum * sum), (sum * sum) / (square * min))
}

/**
 * Lays `items` out in `bounds`, each with an area proportional to its `value`.
 * Items with no value are left out. Largest first, ties in the order given.
 */
export function squarify<T extends { value: number }>(
  items: readonly T[],
  bounds: Rect,
): Placed<T>[] {
  const sorted = items
    .map((item, index) => ({ item, index }))
    .filter(({ item }) => item.value > 0 && Number.isFinite(item.value))
    .sort((a, b) => b.item.value - a.item.value || a.index - b.index)
    .map(({ item }) => item)
  const total = sorted.reduce((sum, item) => sum + item.value, 0)
  if (sorted.length === 0 || bounds.width <= 0 || bounds.height <= 0) return []

  const scale = (bounds.width * bounds.height) / total
  const placed: Placed<T>[] = []
  let { x, y, width, height } = bounds
  let start = 0
  while (start < sorted.length) {
    // The row runs along the shorter side, so it is as thick as the rest allows.
    const horizontal = width >= height
    const side = horizontal ? height : width
    let end = start + 1
    let areas = [sorted[start].value * scale]
    while (end < sorted.length) {
      const next = [...areas, sorted[end].value * scale]
      if (worst(next, side) > worst(areas, side)) break
      areas = next
      end += 1
    }
    const rowArea = areas.reduce((sum, area) => sum + area, 0)
    // The last row takes everything that is left, so rounding cannot leave a seam.
    const thickness =
      end === sorted.length ? (horizontal ? width : height) : rowArea / side
    let offset = 0
    for (let i = start; i < end; i += 1) {
      const length =
        end - start === 1 ? side : (areas[i - start] / rowArea) * side
      placed.push(
        horizontal
          ? {
              item: sorted[i],
              x,
              y: y + offset,
              width: thickness,
              height: length,
            }
          : {
              item: sorted[i],
              x: x + offset,
              y,
              width: length,
              height: thickness,
            },
      )
      offset += length
    }
    if (horizontal) {
      x += thickness
      width -= thickness
    } else {
      y += thickness
      height -= thickness
    }
    start = end
  }
  return placed
}
