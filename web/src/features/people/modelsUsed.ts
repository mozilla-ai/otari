import type { UsageGroupRow } from "@/client"
import { groupModels, type ModelGroup } from "@/features/usage/modelGroups"

/**
 * The models a person used: every named model with tokens or spend, its ids on
 * every provider counted as one (`groupModels`), the costliest first, then the
 * most tokens. The fold of smaller models ("other") is not a model, so it is
 * left out.
 */
export function modelsUsed(rows: readonly UsageGroupRow[]): ModelGroup[] {
  return groupModels(rows)
    .filter((model) => !model.isOther && (model.tokens > 0 || model.cost > 0))
    .sort((a, b) => b.cost - a.cost || b.tokens - a.tokens)
}
