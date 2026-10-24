/**
 * What a budget is applied to, as the Budgets table names it.
 *
 * Two axes decide what an entity is called, and the resource axis wins. An
 * entity narrowed to a provider caps the organization's spend with that
 * provider, so it reads as the provider rather than as the scope carrying it;
 * only an un-narrowed entity reads as its scope. Reading `scope_type` alone is
 * what would print the organization twice and the provider never.
 */

import type { AppliedEntity } from "@/client"

/** The phrases, in the order an admin scans them: widest scope first. */
const GROUP_ORDER = [
  "organization",
  "workspace",
  "org_member",
  "workspace_member",
  "api_token",
  "provider",
] as const

type GroupKey = (typeof GROUP_ORDER)[number]

/** What an entity counts as: the provider it narrows to, else its scope. */
function findGroup(entity: AppliedEntity): GroupKey {
  if (entity.provider_key_id) return "provider"
  return GROUP_ORDER.includes(entity.scope_type as GroupKey)
    ? (entity.scope_type as GroupKey)
    : "organization"
}

/** The plural of a count, where a single entity is worth naming instead. */
function formatCount(count: number, singular: string, plural: string): string {
  return `${count} ${count === 1 ? singular : plural}`
}

/**
 * One group's phrase.
 *
 * A group of one is named where the server names it, because "Acme" is what an
 * admin is looking for and "1 workspace" is what makes them open the row to find
 * out which. A membership or a key carries no name, so it stays a count.
 */
function describeGroup(
  group: GroupKey,
  entities: readonly AppliedEntity[],
): string {
  const count = entities.length
  switch (group) {
    case "organization":
      return `${entities[0].name ?? "Organization"} (organization)`
    case "workspace":
      if (count > 1) return formatCount(count, "workspace", "workspaces")
      return entities[0].name ?? "1 workspace"
    case "provider":
      if (count > 1) return formatCount(count, "provider", "providers")
      return entities[0].provider_key_id ?? "1 provider"
    case "org_member":
      return formatCount(count, "organization member", "organization members")
    case "workspace_member":
      return formatCount(count, "workspace member", "workspace members")
    case "api_token":
      return formatCount(count, "API key", "API keys")
  }
}

/** What an "Applied to" cell reads when a budget applies to nothing. */
export const APPLIED_TO_NOTHING = "Not applied yet"

/**
 * The "Applied to" cell for one budget.
 *
 * Capped at `maxPhrases` with the hidden *entities* counted into the overflow,
 * not the hidden phrases: "+3" has to mean three more things this budget caps,
 * or an admin reads a budget covering a dozen workspaces as covering four.
 */
export function formatAppliedTo(
  entities: readonly AppliedEntity[],
  maxPhrases = 3,
): string {
  if (entities.length === 0) return APPLIED_TO_NOTHING

  const groups = GROUP_ORDER.map((group) => ({
    group,
    members: entities.filter((entity) => findGroup(entity) === group),
  })).filter((entry) => entry.members.length > 0)

  const shown = groups.slice(0, maxPhrases)
  const hidden = groups
    .slice(maxPhrases)
    .reduce((total, entry) => total + entry.members.length, 0)

  const phrases = shown.map((entry) =>
    describeGroup(entry.group, entry.members),
  )
  return hidden > 0 ? `${phrases.join(", ")}, +${hidden}` : phrases.join(", ")
}
