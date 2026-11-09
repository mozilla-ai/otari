import { face } from "./faces"

const TONE_FILL: Record<number, string> = {
  1: "var(--color-chart-cat-1)",
  2: "var(--color-chart-cat-2)",
  3: "var(--color-chart-cat-3)",
  4: "var(--color-chart-cat-4)",
  5: "var(--color-chart-cat-5)",
  6: "var(--color-chart-cat-6)",
  7: "var(--color-chart-cat-7)",
  8: "var(--color-chart-cat-8)",
}

/**
 * One face, filling whatever box the caller gives it. Decorative wherever it
 * renders: the name it stands for is always beside it.
 *
 * The background is the tone mixed into the surface by `--avatar-tint-amount`,
 * which each theme sets, so the square keeps its edge on light and dark.
 */
export function AvatarFace({
  id,
  className = "",
}: {
  id: number
  className?: string
}) {
  const { tone, accent, sides, center } = face(id)
  return (
    <svg
      aria-hidden
      data-face={id}
      viewBox="0 0 7 7"
      shapeRendering="crispEdges"
      className={`block shrink-0 ${className}`}
    >
      <rect
        width="7"
        height="7"
        style={{
          fill: `color-mix(in oklab, ${TONE_FILL[tone]} var(--avatar-tint-amount), var(--color-surface))`,
        }}
      />
      <path d={sides} fill={TONE_FILL[tone]} />
      <path d={center} fill={TONE_FILL[accent]} />
    </svg>
  )
}
