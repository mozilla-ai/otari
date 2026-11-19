import type { UsageGroupRow } from "@/client"

/** The name of the server's fold of its smallest rows, on every cut. */
export const OTHER = "Other"

/** One model in a usage breakdown, however many ids it was requested as. */
export interface ModelGroup {
  /** Stable per model: its group, or its one id where the gateway gave none. */
  id: string
  name: string
  /**
   * Every id the usage recorded the model as, busiest first. A model filter
   * matches an id, not a model, so a filter for this model names them all.
   */
  spellings: string[]
  cost: number
  tokens: number
  requests: number
  /** The server's fold of its smallest rows, which is no one model. */
  isOther: boolean
}

/**
 * A summary's model rows, one per model. Providers name one model differently
 * (`accounts/fireworks/models/deepseek-v4p1-flash-1017`,
 * `deepseek-ai/DeepSeek-V4.1-Flash-1017`) and the gateway keeps a row per
 * name, marking the ones that are the same model with a shared `model_group`
 * and one `label`. The rows arrive busiest first, and so do the groups.
 */
export function groupModels(rows: readonly UsageGroupRow[]): ModelGroup[] {
  const groups = new Map<string, ModelGroup>()
  for (const row of rows) {
    // The fold and unnamed usage keep ids no model's group or id can take,
    // as every cut's rows do.
    const id = row.is_other
      ? "model::other"
      : row.model_group
        ? `model:group:${row.model_group}`
        : row.key === null
          ? "model::none"
          : `model:id:${row.key}`
    const group = groups.get(id)
    if (group) {
      group.cost += row.cost
      group.tokens += row.tokens
      group.requests += row.requests
      if (row.key !== null) group.spellings.push(row.key)
      continue
    }
    groups.set(id, {
      id,
      // An empty id is as unnamed as a missing one.
      name: row.is_other ? OTHER : row.label || row.key || "Unknown model",
      spellings: row.key === null ? [] : [row.key],
      cost: row.cost,
      tokens: row.tokens,
      requests: row.requests,
      isOther: row.is_other ?? false,
    })
  }
  return [...groups.values()]
}
