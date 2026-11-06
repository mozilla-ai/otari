/**
 * The avatar set: eight tones from the categorical chart slots, times nine
 * pixel glyphs, is seventy-two faces. A face is its index, so storing a choice
 * is storing one integer and the set grows at the end without moving anyone.
 */

const GLYPHS = [
  { name: "plus", rows: ["..#..", "..#..", "#####", "..#..", "..#.."] },
  { name: "dice", rows: ["#...#", ".....", "..#..", ".....", "#...#"] },
  { name: "stairs", rows: ["#....", "##...", "###..", "####.", "#####"] },
  { name: "diamond", rows: ["..#..", ".#.#.", "#...#", ".#.#.", "..#.."] },
  { name: "checker", rows: ["#.#.#", ".#.#.", "#.#.#", ".#.#.", "#.#.#"] },
  { name: "hourglass", rows: ["#####", ".###.", "..#..", ".###.", "#####"] },
  { name: "frame", rows: ["#####", "#...#", "#.#.#", "#...#", "#####"] },
  { name: "arrow", rows: ["..#..", ".###.", "#.#.#", "..#..", "..#.."] },
  { name: "cross", rows: ["#...#", ".#.#.", "..#..", ".#.#.", "#...#"] },
] as const

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

export const FACE_COUNT = GLYPHS.length * TONES.length

export interface Face {
  id: number
  /** 1 to 8: the `--color-chart-cat-N` slot the tile is filled with. */
  tone: number
  /** The glyph as an SVG path on a 9-unit box. */
  path: string
  label: string
}

/**
 * Horizontal runs rather than one square per cell, so a glyph is a handful of
 * path commands. The glyph sits on a 9-unit box with two units of margin.
 */
function glyphPath(rows: readonly string[]): string {
  let d = ""
  rows.forEach((row, y) => {
    for (const run of row.matchAll(/#+/g)) {
      const x = run.index ?? 0
      d += `M${x + 2} ${y + 2}h${run[0].length}v1h-${run[0].length}z`
    }
  })
  return d
}

const PATHS = GLYPHS.map((glyph) => glyphPath(glyph.rows))

export function face(id: number): Face {
  const safe = ((Math.trunc(id) % FACE_COUNT) + FACE_COUNT) % FACE_COUNT
  // Tone walks fastest, so faces next to each other in the set differ in
  // colour before they differ in shape.
  const tone = safe % TONES.length
  const glyph = Math.floor(safe / TONES.length)
  return {
    id: safe,
    tone: tone + 1,
    path: PATHS[glyph],
    label: `${TONES[tone]} ${GLYPHS[glyph].name}`,
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
