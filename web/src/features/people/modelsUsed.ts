import type { UsageGroupRow } from "@/client"
import { type ModelGroup, rankedModels } from "@/features/usage/modelGroups"

/**
 * The models a person used: every model with tokens or spend, in the order
 * `rankedModels` gives them.
 */
export function modelsUsed(rows: readonly UsageGroupRow[]): ModelGroup[] {
  return rankedModels(rows).filter(
    (model) => model.tokens > 0 || model.cost > 0,
  )
}
