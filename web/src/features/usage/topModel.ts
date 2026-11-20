import type { UsageGroupRow } from "@/client"
import { wholePct } from "@/features/spend/limits"
import { rankedModels } from "@/features/usage/modelGroups"

export interface TopModel {
  /** The model's name, or "Unknown model" where the usage names none. */
  name: string
  /** Every id the usage recorded it as, which the name stands in for. */
  spellings: string[]
  cost: number
}

/** The model a row spent the most on, as `rankedModels` ranks them. */
export function pickTopModel(
  rows: readonly UsageGroupRow[],
): TopModel | undefined {
  const [top] = rankedModels(rows)
  return top
    ? { name: top.name, spellings: top.spellings, cost: top.cost }
    : undefined
}

/** How much of the row's spend the top model took, or why there is no figure. */
export function topModelShare(top: TopModel, rowCost: number): string {
  return rowCost > 0 ? `${wholePct(top.cost / rowCost)} of spend` : "unpriced"
}
