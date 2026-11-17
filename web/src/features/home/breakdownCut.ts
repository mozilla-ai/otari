import type { ReadState } from "@/features/spend/useSpend"

/**
 * Which cut "Where it went" opens on: the one the URL names, else People for a
 * team and API keys for someone alone. That default is the roster's answer,
 * so there is none while the roster loads; picked early, it would open
 * everyone on API keys and move a team to People a moment later.
 */
export function breakdownCut({
  chosen,
  roster,
  isTeam,
}: {
  chosen: string | undefined
  roster: ReadState
  isTeam: boolean
}): string | undefined {
  if (chosen) return chosen
  if (roster === "loading") return undefined
  return isTeam ? "people" : "keys"
}
