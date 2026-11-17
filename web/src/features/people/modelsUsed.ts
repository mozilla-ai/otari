import type { UsageGroupRow } from "@/client"

/**
 * The models a person used: every named model with tokens or spend, the
 * costliest first, then the most tokens. The fold of smaller models ("other")
 * is not a model, so it is left out.
 */
export function modelsUsed(rows: readonly UsageGroupRow[]): UsageGroupRow[] {
  return rows
    .filter((row) => !row.is_other && (row.tokens > 0 || row.cost > 0))
    .sort((a, b) => b.cost - a.cost || b.tokens - a.tokens)
}
