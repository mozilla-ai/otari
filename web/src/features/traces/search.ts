import type { Observation } from "./types"

// Langfuse's rule (matchesSearchQuery): name, type or id contains the query.
export function matchesSearch(obs: Observation, query: string): boolean {
  const q = query.trim().toLowerCase()
  if (!q) return true
  return [obs.name, obs.type, obs.id].some((v) => v.toLowerCase().includes(q))
}
