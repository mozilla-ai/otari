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
 * The pattern is cut in the surface colour, so it reads white on the light
 * theme and near-black on the dark one, where the chart slots are lighter.
 */
export function AvatarFace({
  id,
  className = "",
}: {
  id: number
  className?: string
}) {
  const { tone, path } = face(id)
  return (
    <svg
      aria-hidden
      viewBox="0 0 7 7"
      shapeRendering="crispEdges"
      className={`block shrink-0 ${className}`}
    >
      <rect width="7" height="7" fill={TONE_FILL[tone]} />
      <path d={path} fill="var(--color-surface)" />
    </svg>
  )
}
