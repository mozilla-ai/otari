import { OBSERVATION_TYPES } from "./observationTypes"
import type { Observation } from "./types"

// A span matches when its name, its type as its row says it or its id contains
// the query, so a pasted id finds its span as readily as a typed name does.
export function matchesSearch(obs: Observation, query: string): boolean {
  const q = query.trim().toLowerCase()
  if (!q) return true
  return [obs.name, OBSERVATION_TYPES[obs.type].label, obs.id].some((v) =>
    v.toLowerCase().includes(q),
  )
}
