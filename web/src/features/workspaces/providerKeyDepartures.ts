import type { WorkspaceProviderKeyOverride } from "@/client"

/**
 * What one workspace changed about its organization's provider keys, in a phrase.
 *
 * The three departures are counted separately, because they cost a workspace
 * different things: a pin only chooses between keys, a narrowing takes models
 * off its catalog, and a disable takes a provider off it. Worded the way the
 * picker in the edit form words them, so the summary and the setting it points
 * at do not use two vocabularies for one state.
 *
 * Undefined when there is nothing to say, which covers both an organization
 * holding no keys and a read that has not answered. The summary is a pointer
 * into the workspace's edit form, so an absent count is a quiet cell rather
 * than a claim either way.
 */
export function departureSummary(
  rows: WorkspaceProviderKeyOverride[] | undefined,
): { text: string; hasDepartures: boolean } | undefined {
  if (rows === undefined || rows.length === 0) return undefined
  const counts = [
    [rows.filter((row) => row.is_default).length, "always used"],
    [rows.filter((row) => row.allowed_models.length > 0).length, "narrowed"],
    [rows.filter((row) => row.disabled).length, "never used"],
  ] as const
  const parts = counts
    .filter(([count]) => count > 0)
    .map(([count, what]) => `${count} ${what}`)
  return parts.length === 0
    ? { text: "Inherits all", hasDepartures: false }
    : { text: parts.join(", "), hasDepartures: true }
}
