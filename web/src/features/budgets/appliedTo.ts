/**
 * What a budget is applied to, as the Budgets table names it.
 *
 * A budget is one object: a limit, a reset cycle, and the entities it applies
 * to. The entities are its ceilings, so this is the derivation that turns a
 * list of ceiling rows into the phrase a single table cell can hold.
 *
 * Two axes decide what an entity is called, and the resource axis wins. A
 * ceiling narrowed to a provider, or to one of its models, caps the
 * organization's spend there, so it reads as the provider or the model rather
 * than as the scope carrying it; only an un-narrowed ceiling reads as its scope. Reading `scope_type` alone is
 * what would print the organization twice and the provider never.
 */

import type { OrganizationSpendCeiling, Workspace } from "@/client"

/** The phrases, in the order an admin scans them: widest scope first. */
const GROUP_ORDER = [
  "organization",
  "workspace",
  "org_member",
  "workspace_member",
  "api_token",
  "provider",
  "model",
] as const

type GroupKey = (typeof GROUP_ORDER)[number]

/** What a ceiling counts as: the model or provider it narrows to, else its scope. */
function groupOf(ceiling: AppliedCeiling): GroupKey {
  if (ceiling.model) return "model"
  if (ceiling.provider_key_id) return "provider"
  return GROUP_ORDER.includes(ceiling.scope_type as GroupKey)
    ? (ceiling.scope_type as GroupKey)
    : "organization"
}

/** The ceiling fields a phrase is derived from, as the list endpoint carries them. */
export type AppliedCeiling = Pick<
  OrganizationSpendCeiling,
  "scope_type" | "scope_id" | "provider_key_id" | "model" | "budget_id"
>

export type AppliedToContext = {
  organizationName: string
  /** Only what a name is resolved from, so a caller holding less can still ask. */
  workspaces: readonly Pick<Workspace, "id" | "name">[]
}

/** The plural of a count, where a single entity is worth naming instead. */
function counted(count: number, singular: string, plural: string): string {
  return `${count} ${count === 1 ? singular : plural}`
}

/**
 * One group's phrase.
 *
 * A group of one is named where a name is reachable, because "Acme" is what an
 * admin is looking for and "1 workspace" is what makes them open the row to find
 * out which. A membership has no name on this page (the ceiling carries an id
 * and the roster is a different read), so it stays a count at every size.
 */
function groupPhrase(
  group: GroupKey,
  ceilings: readonly AppliedCeiling[],
  context: AppliedToContext,
): string {
  const count = ceilings.length
  switch (group) {
    case "organization":
      return `${context.organizationName} (organization)`
    case "workspace": {
      if (count > 1) return counted(count, "workspace", "workspaces")
      const workspace = context.workspaces.find(
        (candidate) => candidate.id === ceilings[0].scope_id,
      )
      return workspace ? workspace.name : "1 workspace"
    }
    case "provider": {
      if (count > 1) return counted(count, "provider", "providers")
      return ceilings[0].provider_key_id ?? "1 provider"
    }
    case "model": {
      if (count > 1) return counted(count, "model", "models")
      // A model id is only unique within its provider, so the provider is named too.
      return `${ceilings[0].model} on ${ceilings[0].provider_key_id}`
    }
    case "org_member":
      return counted(count, "organization member", "organization members")
    case "workspace_member":
      return counted(count, "workspace member", "workspace members")
    case "api_token":
      return counted(count, "API key", "API keys")
  }
}

/** Every ceiling in the list, grouped under the budget it enforces. */
export function ceilingsByBudget<T extends AppliedCeiling>(
  ceilings: readonly T[],
): Map<string, T[]> {
  return ceilings.reduce((grouped, ceiling) => {
    const held = grouped.get(ceiling.budget_id)
    if (held) held.push(ceiling)
    else grouped.set(ceiling.budget_id, [ceiling])
    return grouped
  }, new Map<string, T[]>())
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
export function appliedToLabel(
  ceilings: readonly AppliedCeiling[],
  context: AppliedToContext,
  maxPhrases = 3,
): string {
  if (ceilings.length === 0) return APPLIED_TO_NOTHING

  const groups = GROUP_ORDER.map((group) => ({
    group,
    members: ceilings.filter((ceiling) => groupOf(ceiling) === group),
  })).filter((entry) => entry.members.length > 0)

  const shown = groups.slice(0, maxPhrases)
  const hidden = groups
    .slice(maxPhrases)
    .reduce((total, entry) => total + entry.members.length, 0)

  const phrases = shown.map((entry) =>
    groupPhrase(entry.group, entry.members, context),
  )
  return hidden > 0 ? `${phrases.join(", ")}, +${hidden}` : phrases.join(", ")
}
