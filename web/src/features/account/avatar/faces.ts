/**
 * The avatar set: a mirrored 5x5 pixel pattern in one of the eight categorical
 * chart tones, on a light tint of that tone, with the center column in a
 * second tone. The second tone is fixed by the first (three slots along), so
 * it adds color without adding faces that differ only in a sliver.
 *
 * It is generated rather than drawn, because an avatar's job is telling people
 * apart, and a hand-drawn set repeats: with 72 faces, two people in a team of
 * 30 share one almost certainly. Here there are about 204,000, so the chance in
 * a team of 30 is about 0.2%, and in a team of 200 about 9%.
 *
 * A face is its index, so storing a choice is storing one integer. Tone walks
 * fastest, so neighboring ids differ in color before they differ in shape.
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
 * something in the center column. That rules out slivers, slabs and faces
 * that are two separate halves, and leaves 25,487 patterns.
 */
const PATTERNS: readonly number[] = (() => {
  const kept: number[] = []
  for (let bits = 0; bits < 1 << 15; bits++) {
    let lit = 0
    let rows = 0
    let center = 0
    for (let y = 0; y < 5; y++) {
      const row = (bits >> (y * 3)) & 0b111
      if (row) rows++
      if (row & 0b100) center++
      lit += (row & 1) + ((row >> 1) & 1) + ((row >> 2) & 1)
    }
    if (lit >= 6 && lit <= 10 && rows >= 3 && center >= 1) kept.push(bits)
  }
  return kept
})()

export const FACE_COUNT = PATTERNS.length * TONES.length

export interface Face {
  id: number
  /** 1 to 8: the `--color-chart-cat-N` slot of the pattern and its tint. */
  tone: number
  /** 1 to 8: the slot of the center column. */
  accent: number
  /** The mirrored sides of the pattern, as an SVG path on a 7-unit box. */
  sides: string
  /** The center column of the pattern, on the same box. */
  center: string
  label: string
}

/**
 * The pattern as two paths on a 7-unit box with one unit of margin: the sides
 * (columns 0 and 1, mirrored onto 4 and 3) and the center column.
 */
function patternPaths(bits: number): { sides: string; center: string } {
  let sides = ""
  let center = ""
  for (let y = 0; y < 5; y++) {
    const row = (bits >> (y * 3)) & 0b111
    for (const x of [0, 1]) {
      if (row & (1 << x)) {
        sides += `M${x + 1} ${y + 1}h1v1h-1zM${5 - x} ${y + 1}h1v1h-1z`
      }
    }
    if (row & 0b100) center += `M3 ${y + 1}h1v1h-1z`
  }
  return { sides, center }
}

export function face(id: number): Face {
  const safe = ((Math.trunc(id) % FACE_COUNT) + FACE_COUNT) % FACE_COUNT
  const tone = safe % TONES.length
  const pattern = Math.floor(safe / TONES.length)
  return {
    id: safe,
    tone: tone + 1,
    accent: ((tone + 3) % TONES.length) + 1,
    ...patternPaths(PATTERNS[pattern]),
    // Patterns have no names worth reading out, so the number stands in: it
    // is stable, and it tells two faces of one color apart.
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
