/**
 * The avatar set: a mirrored 5x5 pixel pattern, cut in the surface colour out
 * of a square filled with one of the eight categorical chart tones.
 *
 * It is generated rather than drawn, because an avatar's job is telling people
 * apart, and a hand-drawn set repeats: with 72 faces, two people in a team of
 * 30 share one almost certainly. Here there are about 204,000, so the chance in
 * a team of 30 is about 0.2%, and in a team of 200 about 9%.
 *
 * A face is its index, so storing a choice is storing one integer. Tone walks
 * fastest, so neighbouring ids differ in colour before they differ in shape.
 */

// Named for the hue each chart slot resolves to, in slot order.
const TONES = [
  "teal",
  "copper",
  "violet",
  "green",
  "plum",
  "blue",
  "ochre",
  "brick",
] as const

/**
 * Every pattern worth wearing. A 5x5 grid mirrored left to right has 15 free
 * cells (three columns by five rows); of the 32,768 ways to fill them, these
 * are the ones with 6 to 10 cells lit, spread over at least three rows, with
 * something in the centre column. That rules out slivers, slabs and faces
 * that are two separate halves, and leaves 25,487 patterns.
 */
const PATTERNS: readonly number[] = (() => {
  const kept: number[] = []
  for (let bits = 0; bits < 1 << 15; bits++) {
    let lit = 0
    let rows = 0
    let centre = 0
    for (let y = 0; y < 5; y++) {
      const row = (bits >> (y * 3)) & 0b111
      if (row) rows++
      if (row & 0b100) centre++
      lit += (row & 1) + ((row >> 1) & 1) + ((row >> 2) & 1)
    }
    if (lit >= 6 && lit <= 10 && rows >= 3 && centre >= 1) kept.push(bits)
  }
  return kept
})()

export const FACE_COUNT = PATTERNS.length * TONES.length

export interface Face {
  id: number
  /** 1 to 8: the `--color-chart-cat-N` slot the tile is filled with. */
  tone: number
  /** The pattern as an SVG path on a 7-unit box, one unit of margin around it. */
  path: string
  label: string
}

/** Horizontal runs rather than one square per cell, so a face is a few commands. */
function patternPath(bits: number): string {
  let d = ""
  for (let y = 0; y < 5; y++) {
    const row = (bits >> (y * 3)) & 0b111
    // Columns 0 and 1 mirror onto 4 and 3; column 2 is the centre.
    const lit = [0, 1, 2, 1, 0].map(
      (x) => (x === 2 ? row & 0b100 : row & (1 << x)) !== 0,
    )
    let x = 0
    while (x < 5) {
      if (!lit[x]) {
        x++
        continue
      }
      let end = x
      while (end < 5 && lit[end]) end++
      d += `M${x + 1} ${y + 1}h${end - x}v1h-${end - x}z`
      x = end
    }
  }
  return d
}

export function face(id: number): Face {
  const safe = ((Math.trunc(id) % FACE_COUNT) + FACE_COUNT) % FACE_COUNT
  const tone = safe % TONES.length
  const pattern = Math.floor(safe / TONES.length)
  return {
    id: safe,
    tone: tone + 1,
    path: patternPath(PATTERNS[pattern]),
    // Patterns have no names worth reading out, so the number stands in: it
    // is stable, and it tells two faces of one colour apart.
    label: `${TONES[tone]} pattern ${pattern + 1}`,
  }
}

/**
 * The face someone has before they pick one: drawn at random, but from their
 * user id, so it is the same face on every device and every visit without
 * being stored anywhere. FNV-1a, which spreads ids that share a prefix.
 */
export function defaultFaceFor(seed: string): number {
  let hash = 0x811c9dc5
  for (let i = 0; i < seed.length; i++) {
    hash ^= seed.charCodeAt(i)
    hash = Math.imul(hash, 0x01000193) >>> 0
  }
  return hash % FACE_COUNT
}

/** `count` distinct random faces, for the picker to lay out. */
export function sampleFaces(count: number, random = Math.random): number[] {
  const picked = new Set<number>()
  while (picked.size < count) picked.add(Math.floor(random() * FACE_COUNT))
  return [...picked]
}
