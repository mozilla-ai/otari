/**
 * Who a new spend rule can land on, one list per kind: the organization, a
 * workspace, a person or an API key. Each row carries what the limit dialog
 * needs to write the limit, the words its row and helper line show, and the
 * limit it already has, read once so the row, the helper and the button agree.
 */

import type {
  ApiKey,
  OrganizationMember,
  OrganizationSpendCeiling,
  Workspace,
} from "@/client"
import { keyLabel, keyState } from "@/features/keys/keyList"
import { memberLabel, membershipLabel } from "@/features/organization/roles"
import {
  type AppliedLimit,
  type LimitScope,
  limitsOn,
} from "@/features/spend/limits"
import type { ReadState } from "@/features/spend/useSpend"
import { formatUsd } from "@/shared/helpers/format"

export type TargetKind = "everyone" | "workspace" | "person" | "key"

/** The kinds a picker lists; Everyone is one target, so it has no list. */
type ListedKind = Exclude<TargetKind, "everyone">

interface KindCopy {
  label: string
  one: string
  many: string
  /** What the search reads, after "by". */
  by: string
}

export const KIND_COPY: Record<ListedKind, KindCopy> = {
  workspace: {
    label: "Workspace",
    one: "workspace",
    many: "workspaces",
    by: "name",
  },
  person: {
    label: "Person",
    one: "person",
    many: "people",
    by: "name or email",
  },
  key: {
    label: "API key",
    one: "API key",
    many: "API keys",
    by: "name or owner",
  },
}

export const TARGET_KINDS: { value: TargetKind; label: string }[] = [
  { value: "everyone", label: "Everyone" },
  ...(["workspace", "person", "key"] as const).map((value) => ({
    value,
    label: KIND_COPY[value].label,
  })),
]

export function asTargetKind(value: string): TargetKind | undefined {
  return TARGET_KINDS.find((one) => one.value === value)?.value
}

export interface RuleTarget {
  /** Unique across kinds: the scope and its id. */
  value: string
  kind: TargetKind
  scope: LimitScope
  scopeId: string
  /** Who the limit is on, as the limit dialog and the budget name them. */
  name: string
  /** What tells this row from one with the same name: "ada@x.example · Admin". */
  detail: string
  /** What a search reads besides the name. */
  alsoMatches: string[]
  own: AppliedLimit | null
}

interface TargetSources {
  organization?: { id: string; name: string }
  workspaces: readonly Workspace[]
  members: readonly OrganizationMember[]
  keys: readonly ApiKey[]
  ceilings: readonly OrganizationSpendCeiling[]
  /** When the page opened: a key past its expiry cannot be used, so is not offered. */
  now: number
}

export const SEPARATOR = " · "

const joined = (parts: (string | undefined)[]) =>
  parts.filter(Boolean).join(SEPARATOR)

/** Everything a limit can be set on, in the order the kinds are offered. */
export function buildTargets(sources: TargetSources): RuleTarget[] {
  const { organization, workspaces, members, keys, ceilings, now } = sources
  const active = members.filter((one) => one.status === "active")
  const headcount = new Map<string, number>()
  for (const member of active) {
    for (const place of member.workspaces ?? []) {
      headcount.set(
        place.workspace_id,
        (headcount.get(place.workspace_id) ?? 0) + 1,
      )
    }
  }
  const workspaceName = new Map(workspaces.map((one) => [one.id, one.name]))
  // A key names its owner by either id a member carries.
  const ownerName = new Map(
    members.flatMap((member) =>
      [member.user_id, member.attribution_user_id].flatMap((id) =>
        id ? [[id, memberLabel(member)] as const] : [],
      ),
    ),
  )
  const target = (
    kind: TargetKind,
    scope: LimitScope,
    scopeId: string,
    name: string,
    detail: string,
    alsoMatches: string[] = [],
  ): RuleTarget => ({
    value: `${scope}:${scopeId}`,
    kind,
    scope,
    scopeId,
    name,
    detail,
    alsoMatches,
    own: limitsOn(ceilings, scope, scopeId).own,
  })

  return [
    ...(organization
      ? [
          target(
            "everyone",
            "organization",
            organization.id,
            organization.name,
            "",
          ),
        ]
      : []),
    ...workspaces.map((one) => {
      const count = headcount.get(one.id) ?? 0
      return target(
        "workspace",
        "workspace",
        one.id,
        one.name,
        count === 1 ? "1 person" : `${count} people`,
      )
    }),
    ...active.flatMap((member) => {
      if (!member.organization_member_id) return []
      const name = member.full_name || member.email || "A member"
      const email = member.email && member.email !== name ? member.email : ""
      return [
        target(
          "person",
          "org_member",
          member.organization_member_id,
          name,
          joined([email, membershipLabel(member.role)]),
          member.email ? [member.email] : [],
        ),
      ]
    }),
    ...keys
      .filter((key) => keyState(key, now) === "active")
      .map((key) => {
        const owner = key.user_id ? ownerName.get(key.user_id) : undefined
        return target(
          "key",
          "api_token",
          key.id,
          keyLabel(key),
          joined([owner, workspaceName.get(key.workspace_id)]),
          owner ? [owner] : [],
        )
      }),
  ]
}

export function targetsOfKind(
  targets: readonly RuleTarget[],
  kind: TargetKind,
): RuleTarget[] {
  return targets.filter((one) => one.kind === kind)
}

/** The rows a search keeps: its text anywhere in the name or what stands beside it. */
export function matchTargets(
  targets: readonly RuleTarget[],
  query: string,
): RuleTarget[] {
  const wanted = query.trim().toLowerCase()
  if (!wanted) return [...targets]
  return targets.filter((one) =>
    [one.name, ...one.alsoMatches].some((text) =>
      text.toLowerCase().includes(wanted),
    ),
  )
}

/** "Limit $40.00 a week", or nothing where there is none. */
export function limitWords(own: AppliedLimit | null): string {
  if (!own) return ""
  return `Limit ${formatUsd(own.max)}${own.per ? ` ${own.per}` : ""}`
}

/** The limit a chosen target has, or the lack of one. */
export function limitStatus(
  target: RuleTarget,
  limits: ReadState = "ready",
): string {
  if (limits === "failed") return "Could not load limits"
  return limitWords(target.own) || "No limit yet"
}

/** A row's accessible name: everything it shows, in reading order. */
export function targetRowText(target: RuleTarget): string {
  return [target.name, target.detail, limitWords(target.own)]
    .filter(Boolean)
    .join(", ")
}

/** "Search 12 people by name or email", with the real count. */
export function searchPlaceholder(
  kind: ListedKind,
  count: number,
  read: ReadState,
): string {
  const copy = KIND_COPY[kind]
  if (read === "loading") return `Loading ${copy.many}`
  if (read === "failed") return `Could not load ${copy.many}`
  if (count === 0) return `No ${copy.many} yet`
  return `Search ${count} ${count === 1 ? copy.one : copy.many} by ${copy.by}`
}

/** What the open list says with no rows in it. */
export function emptyListWords(
  kind: ListedKind,
  query: string,
  read: ReadState,
  isSourceEmpty: boolean,
): string {
  const { many } = KIND_COPY[kind]
  if (read === "loading") return `Loading ${many}`
  if (read === "failed") return `Could not load ${many}`
  if (isSourceEmpty) return `No ${many} to choose from yet.`
  return `No ${many} match “${query.trim()}”`
}
