/**
 * One rules registry over the stores that hold rules today, apart: spend
 * ceilings and the budgets behind them, model and tool allow-lists on keys,
 * and the organization's guardrails. Each becomes a row with a type, who it
 * applies to, where it is enforced and how. Hook gates (Actions) live in each
 * repository's `.otari/` directory and no store lists them, so they have no
 * rows.
 *
 * Only owners and admins read the ceilings, budgets and guardrails
 * (`buildRules`). Anyone else reads what the server gives them
 * (`buildMemberRules`): each person's per-workspace ceiling amount on the
 * roster, the per-person default of their own workspaces, and their own keys.
 */

import type {
  ApiKey,
  OrganizationBudget,
  OrganizationGuardrail,
  OrganizationMember,
  OrganizationSpendCeiling,
  Workspace,
  WorkspaceBudgetDefault,
} from "@/client"
import { keyState } from "@/features/keys/keyList"
import { attributionId } from "@/features/organization/attribution"
import { memberLabel } from "@/features/organization/roles"
import type { LimitScope, LimitTarget } from "@/features/spend/limits"
import { limitsOn, perWords, resetEndedPeriods } from "@/features/spend/limits"
import { keyName } from "@/features/spend/names"
import { formatUsd } from "@/shared/helpers/format"

type RuleType = "access" | "spend" | "data"
export type RuleMode = "Enforce" | "Watch" | "Off"

export const RULE_TYPE_LABEL: Record<RuleType, string> = {
  access: "Access",
  spend: "Spend",
  data: "Data",
}

/**
 * The views of the list. Actions has none: nothing lists hook gates, so its
 * tab could only hold a sentence, which the All view carries under the table.
 */
export type RuleTab = "all" | "spend" | "access" | "data"

export const RULE_TAB_LABEL: Record<RuleTab, string> = {
  all: "All",
  spend: "Spend",
  access: "Access",
  data: "Data",
}

const TAB_TYPE: Record<Exclude<RuleTab, "all">, RuleType> = {
  spend: "spend",
  access: "access",
  data: "data",
}

/** Guardrails are refused to anyone but owners and admins, so is their tab. */
export function ruleTabs(isManager: boolean): RuleTab[] {
  return isManager
    ? ["all", "spend", "access", "data"]
    : ["all", "spend", "access"]
}

/** The tab a `?type=` value opens, or All for one this caller has not got. */
export function ruleTabOf(value: string, isManager: boolean): RuleTab {
  return ruleTabs(isManager).find((tab) => tab === value) ?? "all"
}

export function rulesOnTab(rules: readonly Rule[], tab: RuleTab): Rule[] {
  return tab === "all"
    ? [...rules]
    : rules.filter((rule) => rule.type === TAB_TYPE[tab])
}

/** Whose requests a rule reaches, as the usage log filters them. */
export interface RuleSubject {
  userIds?: string[]
  apiKeyIds?: string[]
  workspaceId?: string
  /** The whole organization, read unfiltered; no id at all is nobody. */
  everyone?: true
}

/** Who a rule reaches, as the Applies to cell shows it. */
export type RuleScope =
  | { kind: "everyone"; organization?: string }
  | { kind: "team"; team: string }
  | { kind: "teams"; teams: string[] }
  | { kind: "each"; teams: string[]; count: number; includesYou: boolean }
  | {
      kind: "person"
      name: string
      avatarUserId?: string
      avatarFace?: number | null
    }
  /** A key the caller cannot read has no name to show. */
  | { kind: "key"; name?: string }

/** One person a per-person rule reaches, on the roster today. */
export interface RulePerson {
  name: string
  /** The id the usage log and sessions use. */
  userId: string | null
  /** Their login id, which picks their avatar. */
  avatarUserId?: string
  /** The face they picked, if any. */
  avatarFace?: number | null
  /** Their membership, which their person page is under. */
  memberId?: string
  team: string
  /** Their own count against the limit, all their workspaces together; owners and admins only. */
  spent?: number
  /**
   * The placement closest to its own ceiling. Each workspace enforces its own,
   * so what they have left is that one's, not the sum's.
   */
  peak?: { spent: number; max: number }
  isYou: boolean
}

interface SpendFigures {
  max: number
  /** "a week", "a month", "every 10 days", or "" where it is not known. */
  per: string
  /** A limit each person has to themselves, rather than one shared. */
  isEach: boolean
  /** Spent this period; for a per-person rule, everyone's together. Owners and admins only. */
  spent?: number
  /** Spent over the limit; for a per-person rule, the highest placement's share. */
  used?: number
  /** The spend `used` is the share of: the limit's, or the highest placement's. */
  usedSpent?: number
  /** The ceiling `usedSpent` is counted against, where it is not `max`. */
  usedMax?: number
  /** What the limit dialog changes, where this page can change it. */
  target?: LimitTarget
}

/** A workspace's per-person default, where a per-person rule is one. */
export interface TeamDefault {
  workspaceId: string
  team: string
  policyId: string
  budgetId: string
  max: number | null
  /** How often each person's limit resets. */
  period: Pick<
    WorkspaceBudgetDefault,
    "reset_alignment" | "budget_duration_sec"
  >
}

/**
 * How much of a limit is spent, as `SpendMeter` reads it: against a $0 limit,
 * any spend is all of it.
 */
export function shareOf(spent: number, max: number): number {
  return max > 0 ? spent / max : spent > 0 ? 1 : 0
}

export interface Rule {
  id: string
  name: string
  type: RuleType
  scope: RuleScope
  mode: RuleMode
  subject: RuleSubject
  spend?: SpendFigures
  /** For an access rule: the lists, or null where any is allowed. */
  access?: { models: string[] | null; tools: string[] | null }
  /** Everyone a per-person rule reaches who is still in the organization. */
  people?: RulePerson[]
  /** People who left and still carry the rule's ceiling. */
  formerCount?: number
  teamDefault?: TeamDefault
  /** What the control plane stores for the rule, where the caller may read it. */
  definition?: unknown
  createdAt?: string
  updatedAt?: string
}

export interface WorkspaceDefaultRow {
  workspaceId: string
  default: WorkspaceBudgetDefault
}

export interface RuleSources {
  ceilings: readonly OrganizationSpendCeiling[]
  budgets: readonly Pick<OrganizationBudget, "budget_id" | "name">[]
  members: readonly OrganizationMember[]
  keys: readonly ApiKey[]
  workspaces: readonly Workspace[]
  guardrails: readonly OrganizationGuardrail[]
  defaults: readonly WorkspaceDefaultRow[]
  organizationName?: string
  /** When the spend is read: a period that ended before it reads as nothing spent. */
  now: number
}

interface MemberRuleSources
  extends Pick<RuleSources, "members" | "keys" | "defaults" | "now"> {
  /** The caller's membership, so the rows they are under can say so. */
  callerMemberId?: string
}

/** The rule a ceiling belongs to: a per-person ceiling is one of its budget's. */
export function ruleIdForCeiling(
  ceiling: Pick<OrganizationSpendCeiling, "id" | "scope_type" | "budget_id">,
): string {
  return ceiling.scope_type === "workspace_member"
    ? `spend-each-${ceiling.budget_id}`
    : `spend-${ceiling.id}`
}

const defaultRowOf = (
  defaults: readonly WorkspaceDefaultRow[],
  budgetId: string,
) =>
  defaults.find(
    (one) =>
      one.default.budget_id === budgetId &&
      one.default.provider_key_id === null,
  )

/** The default a budget is, if a workspace hands it to each person. */
function defaultFor(
  defaults: readonly WorkspaceDefaultRow[],
  budgetId: string,
  teamName: (workspaceId: string) => string,
): TeamDefault | undefined {
  const row = defaultRowOf(defaults, budgetId)
  if (!row) return undefined
  return {
    workspaceId: row.workspaceId,
    team: teamName(row.workspaceId),
    policyId: row.default.id,
    budgetId,
    max: row.default.max_budget,
    period: {
      reset_alignment: row.default.reset_alignment,
      budget_duration_sec: row.default.budget_duration_sec,
    },
  }
}

/** A key that is disabled or past its expiry is refused whatever it allows, so it enforces nothing. */
function keyAccessRules(keys: readonly ApiKey[], now: number): Rule[] {
  return keys.flatMap((key): Rule[] => {
    if (
      keyState(key, now) !== "active" ||
      (!key.allowed_models && !key.allowed_tools)
    ) {
      return []
    }
    const name = keyName(key)
    const scope: RuleScope = { kind: "key", name }
    return [
      {
        id: `access-key-${key.id}`,
        name: `${name}: allowed models and tools`,
        type: "access",
        scope,
        mode: "Enforce",
        subject: { apiKeyIds: [key.id] },
        access: { models: key.allowed_models, tools: key.allowed_tools },
        definition: {
          allowed_models: key.allowed_models,
          allowed_tools: key.allowed_tools,
        },
        createdAt: key.created_at,
      },
    ]
  })
}

interface Placement {
  member: OrganizationMember
  workspaceId: string
  team: string
  /** This placement's own count, where the caller may read it. */
  spent?: number
  /** This placement's own ceiling, where the caller may read it. */
  max?: number
}

/** One row per person, their teams joined, in name order. */
function peopleOf(
  placements: readonly Placement[],
  callerMemberId: string | undefined,
): RulePerson[] {
  const byMember = new Map<string, RulePerson>()
  for (const placement of placements) {
    const { member } = placement
    const key = member.organization_member_id ?? memberLabel(member)
    const spent = placement.spent
    const peak =
      spent !== undefined && placement.max !== undefined
        ? { spent, max: placement.max }
        : undefined
    const seen = byMember.get(key)
    if (seen) {
      seen.team = `${seen.team}, ${placement.team}`
      if (spent !== undefined) seen.spent = (seen.spent ?? 0) + spent
      if (
        peak &&
        (!seen.peak ||
          shareOf(peak.spent, peak.max) >
            shareOf(seen.peak.spent, seen.peak.max))
      ) {
        seen.peak = peak
      }
      continue
    }
    byMember.set(key, {
      name: memberLabel(member),
      userId: attributionId(member),
      avatarUserId: member.user_id ?? undefined,
      avatarFace: member.avatar_face,
      memberId: member.organization_member_id ?? undefined,
      team: placement.team,
      spent,
      peak,
      isYou:
        callerMemberId !== undefined &&
        member.organization_member_id === callerMemberId,
    })
  }
  return [...byMember.values()].sort((a, b) => a.name.localeCompare(b.name))
}

const uniqueTeams = (placements: readonly Placement[]) => [
  ...new Set(placements.map((one) => one.team)),
]

/** Every rule, as an owner or admin reads them. */
export function buildRules(sources: RuleSources): Rule[] {
  const ceilings = resetEndedPeriods(sources.ceilings, sources.now)
  const workspaceName = new Map(
    sources.workspaces.map((one) => [one.id, one.name]),
  )
  const teamName = (id: string) => workspaceName.get(id) ?? "A workspace"
  const budgetName = new Map(
    sources.budgets.map((one) => [one.budget_id, one.name]),
  )
  const active = sources.members.filter((one) => one.status === "active")
  const byMemberId = new Map(
    active.flatMap((member) =>
      member.organization_member_id
        ? [[member.organization_member_id, member] as const]
        : [],
    ),
  )
  const byPlacement = new Map(
    active.flatMap((member) =>
      (member.workspaces ?? []).map(
        (place) =>
          [
            place.workspace_member_id,
            {
              member,
              workspaceId: place.workspace_id,
              team: place.workspace_name,
            },
          ] as const,
      ),
    ),
  )
  const keyById = new Map(sources.keys.map((key) => [key.id, key]))

  const rules: Rule[] = []
  const perPersonGroups = new Map<string, OrganizationSpendCeiling[]>()

  for (const ceiling of ceilings) {
    // A $0 limit is a limit: it refuses every request.
    if (typeof ceiling.max_budget !== "number") continue
    if (ceiling.scope_type === "workspace_member") {
      const group = perPersonGroups.get(ceiling.budget_id) ?? []
      group.push(ceiling)
      perPersonGroups.set(ceiling.budget_id, group)
      continue
    }
    const name =
      ceiling.name || budgetName.get(ceiling.budget_id) || "Spend limit"
    let scope: RuleScope = { kind: "everyone" }
    let subject: RuleSubject = {}
    let targetName = name
    let limitScope: LimitScope | undefined
    if (ceiling.scope_type === "organization") {
      limitScope = "organization"
      scope = { kind: "everyone", organization: sources.organizationName }
      subject = { everyone: true }
      targetName = sources.organizationName ?? "the organization"
    } else if (ceiling.scope_type === "workspace") {
      limitScope = "workspace"
      const team = teamName(ceiling.scope_id)
      scope = { kind: "team", team }
      subject = { workspaceId: ceiling.scope_id }
      targetName = team
    } else if (ceiling.scope_type === "org_member") {
      limitScope = "org_member"
      const member = byMemberId.get(ceiling.scope_id)
      const who = member ? memberLabel(member) : "A former member"
      scope = {
        kind: "person",
        name: who,
        avatarUserId: member?.user_id ?? undefined,
        avatarFace: member?.avatar_face,
      }
      const userId = member ? attributionId(member) : null
      subject = userId ? { userIds: [userId] } : {}
      targetName = who
    } else if (ceiling.scope_type === "api_token") {
      limitScope = "api_token"
      const listed = keyById.get(ceiling.scope_id)
      const key = listed ? keyName(listed) : undefined
      scope = { kind: "key", name: key }
      subject = { apiKeyIds: [ceiling.scope_id] }
      targetName = key ?? "this API key"
    }
    rules.push({
      id: ruleIdForCeiling(ceiling),
      name,
      type: "spend",
      scope,
      mode: "Enforce",
      subject,
      spend: {
        max: ceiling.max_budget,
        spent: ceiling.current_spend,
        used: shareOf(ceiling.current_spend, ceiling.max_budget),
        usedSpent: ceiling.current_spend,
        per: perWords(ceiling),
        isEach: false,
        target:
          limitScope && ceiling.provider_key_id === null
            ? {
                scope: limitScope,
                scopeId: ceiling.scope_id,
                name: targetName,
                own: limitsOn(ceilings, limitScope, ceiling.scope_id).own,
              }
            : undefined,
      },
      definition: ceiling,
      createdAt: ceiling.created_at,
      updatedAt: ceiling.updated_at,
    })
  }

  for (const [budgetId, group] of perPersonGroups) {
    const [first] = group
    if (typeof first?.max_budget !== "number") continue
    const max = first.max_budget
    const placed = group.flatMap((one): Placement[] => {
      const found = byPlacement.get(one.scope_id)
      return found
        ? [{ ...found, spent: one.current_spend, max: one.max_budget ?? max }]
        : []
    })
    const people = peopleOf(placed, undefined)
    const scope: RuleScope = {
      kind: "each",
      teams: uniqueTeams(placed),
      count: people.length,
      includesYou: people.some((one) => one.isYou),
    }
    const highest = people.reduce<{ spent: number; max: number } | undefined>(
      (top, one) =>
        one.peak &&
        (!top ||
          shareOf(one.peak.spent, one.peak.max) > shareOf(top.spent, top.max))
          ? one.peak
          : top,
      undefined,
    )
    rules.push({
      id: ruleIdForCeiling(first),
      name: budgetName.get(budgetId) || first.name || "Spend limit per person",
      type: "spend",
      scope,
      mode: "Enforce",
      subject: {
        userIds: people.flatMap((one) => (one.userId ? [one.userId] : [])),
      },
      spend: {
        max,
        per: perWords(first),
        isEach: true,
        spent: people.reduce((sum, one) => sum + (one.spent ?? 0), 0),
        used: highest ? shareOf(highest.spent, highest.max) : 0,
        usedSpent: highest?.spent ?? 0,
        usedMax: highest?.max ?? max,
      },
      people,
      formerCount: group.length - placed.length,
      teamDefault: defaultFor(sources.defaults, budgetId, teamName),
      definition: group,
      createdAt: first.created_at,
      updatedAt: first.updated_at,
    })
  }

  rules.push(...keyAccessRules(sources.keys, sources.now))

  for (const guardrail of sources.guardrails) {
    const scope: RuleScope = guardrail.applies_to_all_workspaces
      ? { kind: "everyone" }
      : { kind: "teams", teams: guardrail.workspace_ids.map(teamName) }
    rules.push({
      id: `data-${guardrail.id}`,
      name: guardrail.profile,
      type: "data",
      scope,
      mode: !guardrail.enabled
        ? "Off"
        : guardrail.mode === "block"
          ? "Enforce"
          : "Watch",
      subject: {},
      definition: guardrail,
      createdAt: guardrail.created_at,
      updatedAt: guardrail.updated_at,
    })
  }

  return rules.sort((a, b) => a.name.localeCompare(b.name))
}

/**
 * The rules a member or viewer may read: one per budget the roster's
 * per-workspace ceilings point at, named and timed where it is a default of a
 * workspace they are in, and their own keys' allow-lists. How much of any
 * limit is spent is not given to them, so none is shown.
 */
export function buildMemberRules(sources: MemberRuleSources): Rule[] {
  const groups = new Map<string, { max: number; placements: Placement[] }>()
  const teamName = new Map<string, string>()
  for (const member of sources.members) {
    if (member.status !== "active") continue
    for (const place of member.workspaces ?? []) {
      teamName.set(place.workspace_id, place.workspace_name)
      const ceiling = place.ceiling
      if (typeof ceiling?.max_budget !== "number") continue
      const group = groups.get(ceiling.budget_id) ?? {
        max: ceiling.max_budget,
        placements: [],
      }
      group.placements.push({
        member,
        workspaceId: place.workspace_id,
        team: place.workspace_name,
      })
      groups.set(ceiling.budget_id, group)
    }
  }

  const rules: Rule[] = []
  for (const [budgetId, { max, placements }] of groups) {
    const teams = uniqueTeams(placements)
    const people = peopleOf(placements, sources.callerMemberId)
    const policy = defaultRowOf(sources.defaults, budgetId)?.default
    const scope: RuleScope = {
      kind: "each",
      teams,
      count: people.length,
      includesYou: people.some((one) => one.isYou),
    }
    rules.push({
      id: `spend-each-${budgetId}`,
      name:
        policy?.name ?? `Per person in ${teams.join(", ")}, ${formatUsd(max)}`,
      type: "spend",
      scope,
      mode: "Enforce",
      subject: {
        userIds: people.flatMap((one) => (one.userId ? [one.userId] : [])),
      },
      spend: { max, per: policy ? perWords(policy) : "", isEach: true },
      people,
      teamDefault: defaultFor(
        sources.defaults,
        budgetId,
        (id) => teamName.get(id) ?? "A workspace",
      ),
    })
  }

  rules.push(...keyAccessRules(sources.keys, sources.now))
  return rules.sort((a, b) => a.name.localeCompare(b.name))
}

type RuleSortColumn = "name" | "limit" | "spent" | "used"

export interface RuleSort {
  column: RuleSortColumn
  direction: "ascending" | "descending"
}

/** Most used first for whoever sees spend; by name for anyone else. */
export function defaultRuleSort(isManager: boolean): RuleSort {
  return isManager
    ? { column: "used", direction: "descending" }
    : { column: "name", direction: "ascending" }
}

const SORT_VALUE: Record<
  Exclude<RuleSortColumn, "name">,
  (rule: Rule) => number | undefined
> = {
  limit: (rule) => rule.spend?.max,
  spent: (rule) => rule.spend?.spent,
  used: (rule) => rule.spend?.used,
}

/**
 * The list in a header's order. Rows with nothing in the column (an access or
 * data rule has no spend) follow the rest by name, whichever way it runs.
 */
export function sortRules(rules: readonly Rule[], sort: RuleSort): Rule[] {
  const sign = sort.direction === "ascending" ? 1 : -1
  const byName = (a: Rule, b: Rule) => a.name.localeCompare(b.name)
  if (sort.column === "name") {
    return [...rules].sort((a, b) => sign * byName(a, b))
  }
  const value = SORT_VALUE[sort.column]
  return [...rules].sort((a, b) => {
    const left = value(a)
    const right = value(b)
    if (left === undefined || right === undefined) {
      if (left === right) return byName(a, b)
      return left === undefined ? 1 : -1
    }
    return sign * (left - right) || byName(a, b)
  })
}

export type RuleAction =
  | "change-limit"
  | "remove-limit"
  | "change-default"
  | "stop-default"

/**
 * What the caller may do to a rule from its row, as the server allows it.
 * Owners and admins change or remove a limit the page can write, and change a
 * workspace's default. An owner or admin of a workspace who is only a member
 * of the organization may stop that workspace's default (production lets them
 * point it or delete it, never set an amount); nobody else gets a menu.
 */
export function ruleActions(
  rule: Rule,
  caller: {
    isManager: boolean
    canManageWorkspace: (workspaceId: string) => boolean
  },
): RuleAction[] {
  if (caller.isManager) {
    if (rule.spend?.target) return ["change-limit", "remove-limit"]
    return rule.teamDefault ? ["change-default"] : []
  }
  return rule.teamDefault &&
    caller.canManageWorkspace(rule.teamDefault.workspaceId)
    ? ["stop-default"]
    : []
}

export type RuleColumn =
  | "name"
  | "type"
  | "scope"
  | "limit"
  | "spent"
  | "used"
  | "models"
  | "tools"
  | "mode"
  | "actions"

/**
 * The columns a tab shows this caller. Spent and Used only where the server
 * gives the spend; the actions lane wherever any row can have a menu.
 */
export function ruleColumns(
  tab: RuleTab,
  isManager: boolean,
  hasActions: boolean,
): RuleColumn[] {
  const actions: RuleColumn[] = hasActions ? ["actions"] : []
  if (tab === "access") {
    return ["name", "scope", "models", "tools", "mode", ...actions]
  }
  if (tab === "data") return ["name", "scope", "mode", ...actions]
  const type: RuleColumn[] = tab === "all" ? ["type"] : []
  const spend: RuleColumn[] = isManager ? ["spent", "used"] : []
  return ["name", ...type, "scope", "limit", ...spend, "mode", ...actions]
}

interface EmptyCopy {
  title: string
  body?: string
  /** Where the body's last word sends the reader. */
  link?: { label: string; to: string }
}

/** Whose API keys the caller reads, which an empty Access tab speaks for. */
interface KeyReader {
  /** The deployment's operator reads every key; anyone else, their own. */
  isDeploymentWide: boolean
  canCreateKey: boolean
  organizationName?: string
}

export function emptyCopy(
  tab: RuleTab,
  isManager: boolean,
  keys: KeyReader = { isDeploymentWide: true, canCreateKey: true },
): EmptyCopy {
  if (tab === "access") {
    // An owner or admin also reads the organization's service keys.
    const whose = keys.isDeploymentWide
      ? "No API key has"
      : isManager
        ? `None of your API keys or ${keys.organizationName ?? "the organization"}’s service keys has`
        : "None of your API keys has"
    return {
      title: `${whose} a list of allowed models or tools.`,
      body: `A key without one can use every model and tool its workspace offers.${
        keys.canCreateKey
          ? " Pick them under More options when you create a key."
          : ""
      }`,
    }
  }
  if (tab === "data") {
    return {
      title: "No guardrails yet.",
      body: "A guardrail checks each request before it reaches a model. Set one up in",
      link: { label: "Guardrails", to: "/organization/guardrails" },
    }
  }
  if (!isManager) {
    return tab === "spend"
      ? { title: "No spend limit reaches your workspaces." }
      : {
          title:
            "No spend limit or model list reaches your workspaces or your API keys.",
        }
  }
  return tab === "spend"
    ? { title: "No spend limits yet.", body: "Set one with New rule." }
    : {
        title: "No rules yet.",
        body: "Set a spend limit for everyone, a workspace, a person or an API key with New rule.",
      }
}

/** What stopping a workspace's default leaves, said for its headcount. */
export function stopDefaultWords(count: number): string {
  const kept =
    count === 1
      ? "The 1 person under it keeps theirs."
      : `The ${count} people under it keep theirs.`
  return `${kept} A new amount is set by an owner or admin.`
}

const PEOPLE_SHOWN = 10

/**
 * The people a rule page lists: most spent first where the spend is shown,
 * otherwise by name, the first ten unless every one was asked for.
 */
export function peopleShown(
  people: readonly RulePerson[],
  isManager: boolean,
  isAll: boolean,
): RulePerson[] {
  const ordered = isManager
    ? [...people].sort(
        (a, b) =>
          (b.spent ?? 0) - (a.spent ?? 0) || a.name.localeCompare(b.name),
      )
    : [...people]
  return isAll ? ordered : ordered.slice(0, PEOPLE_SHOWN)
}

/** The people band's sentence, with those who left counted rather than listed. */
export function peopleBandWords(formerCount: number): string {
  const base = "Each person has the whole limit to themselves."
  if (formerCount <= 0) return base
  return formerCount === 1
    ? `${base} 1 person who has left still carries it and is not listed.`
    : `${base} ${formerCount} people who have left still carry it and are not listed.`
}

/** "2 models, 1 tool"; a list that is not set allows any. */
export function accessWords(access: NonNullable<Rule["access"]>): string {
  const count = (list: string[] | null, one: string, many: string) =>
    list === null
      ? `any ${one}`
      : `${list.length} ${list.length === 1 ? one : many}`
  const words = `${count(access.models, "model", "models")}, ${count(access.tools, "tool", "tools")}`
  return words.charAt(0).toUpperCase() + words.slice(1)
}
