import type { UsageGroupRow } from "@/client"
import { wholePct } from "@/features/spend/limits"
import { groupModels } from "@/features/usage/modelGroups"

export interface TopModel {
  /** The model's name, or "Unknown model" where the usage names none. */
  name: string
  /** Every id the usage recorded it as, which the name stands in for. */
  spellings: string[]
  cost: number
}

/**
 * The model a row spent the most on, its ids on every provider counted as one
 * (`groupModels`). Ranked by spend, then by tokens, and the fold of smaller
 * models ("other") is not a model, so it never wins.
 */
export function pickTopModel(
  rows: readonly UsageGroupRow[],
): TopModel | undefined {
  const [top] = groupModels(rows)
    .filter((model) => !model.isOther)
    .sort((a, b) => b.cost - a.cost || b.tokens - a.tokens)
  return top
    ? { name: top.name, spellings: top.spellings, cost: top.cost }
    : undefined
}

/** How much of the row's spend the top model took, or why there is no figure. */
export function topModelShare(top: TopModel, rowCost: number): string {
  return rowCost > 0 ? `${wholePct(top.cost / rowCost)} of spend` : "unpriced"
}
