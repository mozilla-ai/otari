import { makerMark, providerMark } from "@/shared/helpers/brandMarks"

/**
 * One mark, drawn from the geometry table.
 *
 * Its own module, and the only thing that imports `brandMarks`, so the 70 kB of
 * path data lands in a chunk of its own that `BrandMark` fetches after paint.
 * A default export because `React.lazy` takes one, which is the sanctioned
 * exception to this codebase's named-exports rule.
 *
 * Decorative: a mark is never shown without the name it belongs to, so a label
 * here would read the company out twice.
 */
export default function BrandMarkGlyph({
  markKey,
  kind,
  box,
}: {
  markKey: string
  kind: "provider" | "maker"
  box: string
}) {
  const glyph = kind === "provider" ? providerMark(markKey) : makerMark(markKey)
  // The caller already asked whether a mark exists before rendering this, so
  // reaching here without one means the key lists and the tables disagree.
  // Nothing is drawn rather than throwing: the slot stays the size it was.
  if (glyph === undefined) return null

  const shapes = glyph.shapes.map((shape) => (
    <path key={shape.d} d={shape.d} fillOpacity={shape.fillOpacity} />
  ))
  return (
    <svg
      viewBox={glyph.viewBox}
      aria-hidden="true"
      focusable="false"
      className={`${box} shrink-0 fill-current`}
    >
      {glyph.transform === undefined ? (
        shapes
      ) : (
        <g transform={glyph.transform}>{shapes}</g>
      )}
    </svg>
  )
}
