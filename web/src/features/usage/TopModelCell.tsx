import { Skeleton } from "@/design-system/feedback/Skeleton"
import { topModelShare } from "@/features/usage/topModel"
import type { TopModelState } from "@/features/usage/useTopModels"

/**
 * A row's top model and its share of the row's spend: a lane of the tables
 * that break spend down by person or key.
 */
export function TopModelCell({
  state,
  rowCost,
  isInline,
}: {
  state: TopModelState | undefined
  rowCost: number
  /** On a card the model and its share share a line, and may wrap. */
  isInline: boolean
}) {
  if (!state || state.status === "none") {
    return <span className="text-muted">—</span>
  }
  if (state.status === "loading") {
    return <Skeleton className="h-[0.8125rem] w-28" />
  }
  if (state.status === "failed" || !state.top) {
    return <span className="text-caption">Not loaded</span>
  }
  const { top } = state
  return (
    <span
      className={
        isInline
          ? "flex flex-wrap items-baseline gap-x-2"
          : "flex min-w-0 flex-col gap-0.5"
      }
    >
      <span
        className={`text-foreground [overflow-wrap:anywhere] ${isInline ? "" : "line-clamp-1"}`}
        title={top.spellings.join("\n") || top.name}
      >
        {top.name}
      </span>
      <span className="text-caption">{topModelShare(top, rowCost)}</span>
    </span>
  )
}
