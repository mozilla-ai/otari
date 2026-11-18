import type { UsageGroupRow } from "@/client"
import { wholePct } from "@/features/spend/limits"

export interface TopModel {
  /** The model's id, or "Unknown model" where the usage names none. */
  name: string
  cost: number
}

/**
 * The model a row spent the most on. Ranked by spend, then by tokens, and the
 * fold of smaller models ("other") is not a model, so it never wins.
 */
export function pickTopModel(
  rows: readonly UsageGroupRow[],
): TopModel | undefined {
  const [top] = rows
    .filter((row) => !row.is_other)
    .sort((a, b) => b.cost - a.cost || b.tokens - a.tokens)
  return top ? { name: top.key ?? "Unknown model", cost: top.cost } : undefined
}

/** How much of the row's spend the top model took, or why there is no figure. */
export function topModelShare(top: TopModel, rowCost: number): string {
  return rowCost > 0 ? `${wholePct(top.cost / rowCost)} of spend` : "unpriced"
}
