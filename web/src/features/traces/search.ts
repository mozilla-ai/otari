import type { Observation } from "./types"

// A span matches when its name, type or id contains the query, so a pasted id
// finds its span as readily as a typed name does.
export function matchesSearch(obs: Observation, query: string): boolean {
  const q = query.trim().toLowerCase()
  if (!q) return true
  return [obs.name, obs.type, obs.id].some((v) => v.toLowerCase().includes(q))
}
