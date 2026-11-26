import {
  type AttentionInput,
  type AttentionResult,
  attention,
} from "@/features/home/attention"
import { useHomeAccount } from "@/features/home/overlayHomeAccount"
import { type LimitTarget, perWords } from "@/features/spend/limits"
import { memberName } from "@/features/spend/names"
import { toMinute } from "@/features/spend/period"
import type { Spend } from "@/features/spend/useSpend"
import { readState } from "@/features/spend/useSpend"
import { useKeys } from "@/shared/api/apiKeys"
import { useAllSpendCeilings } from "@/shared/api/budgets"
import { useOrganizationMembers } from "@/shared/api/organizations"
import { useRefusalsSince, useUsageSummary } from "@/shared/api/usage"
import { useWorkspaces } from "@/shared/api/workspaces"

const DAY_MS = 86_400_000

/**
 * Reads what the band needs, for the whole organization and the last 24 hours
 * whatever window or workspace the page is showing, and hands it to
 * `attention`. The reads share their cache entries with the rest of Home.
 */
export function useTeamAttention(spend: Spend, now: number): AttentionResult {
  const members = useOrganizationMembers()
  const keys = useKeys()
  const workspaces = useWorkspaces()
  const account = useHomeAccount()
  const ceilings = useAllSpendCeilings()
  const since = new Date(toMinute(now) - DAY_MS).toISOString()
  const day = useUsageSummary(
    { start_date: since },
    "hour",
    ["status_code"],
    true,
    "organization",
  )
  const refused = useRefusalsSince(since)

  const subjects = new Map<string, string>()
  const peopleOf = new Map<string, string[]>()
  const nameByUser = new Map<string, string>()
  for (const member of members.data ?? []) {
    const name = memberName(member)
    const userIds = [member.attribution_user_id, member.user_id].filter(
      (id): id is string => Boolean(id),
    )
    for (const id of userIds) nameByUser.set(id, name)
    if (member.organization_member_id) {
      subjects.set(member.organization_member_id, name)
      peopleOf.set(member.organization_member_id, userIds)
    }
    for (const place of member.workspaces ?? []) {
      subjects.set(
        place.workspace_member_id,
        `${name}, in ${place.workspace_name},`,
      )
      peopleOf.set(place.workspace_member_id, userIds)
    }
  }
  for (const key of keys.data ?? []) {
    subjects.set(
      key.id,
      key.key_name ? `The key ${key.key_name}` : "An unnamed key",
    )
  }
  const workspaceNames = new Map<string, string>()
  for (const workspace of workspaces.data ?? []) {
    subjects.set(workspace.id, `The ${workspace.name} workspace`)
    workspaceNames.set(workspace.id, workspace.name)
  }

  // The row a ceiling sits on, which is what its dialog edits.
  const targets = new Map<string, LimitTarget>()
  for (const row of [...spend.people, ...spend.keys]) {
    const own = row.limits.own
    if (row.target && own) targets.set(own.ceiling.id, row.target)
  }
  const organization = spend.organization
  if (organization.target && organization.limits.own) {
    targets.set(organization.limits.own.ceiling.id, organization.target)
  }
  const list = ceilings.data ?? []
  for (const ceiling of list) {
    const name = workspaceNames.get(ceiling.scope_id)
    const max = ceiling.max_budget
    if (ceiling.scope_type !== "workspace" || !name || max == null) continue
    targets.set(ceiling.id, {
      scope: "workspace",
      scopeId: ceiling.scope_id,
      name,
      own: {
        ceiling,
        max,
        spent: ceiling.current_spend,
        per: perWords(ceiling),
      },
    })
  }

  const states: AttentionInput["states"] = {
    limits:
      readState(ceilings) === "ready" &&
      (members.isLoading || keys.isLoading || workspaces.isLoading)
        ? "loading"
        : readState(ceilings),
    refusals: readState(refused),
    failures: readState(day),
    account: account.state,
  }

  return attention({
    now,
    states,
    ceilings: list,
    subjects,
    workspaceNames,
    targets,
    peopleOf,
    account,
    day: {
      requests: day.data?.totals.request_count ?? 0,
      errors: (day.data?.errors_by_status_code ?? []).map((row) => ({
        errorClass: row.error_class,
        requests: row.requests,
      })),
    },
    refusals: (refused.data?.by_user ?? []).map((row) => ({
      userId: row.is_other ? null : row.key,
      name:
        (row.key ? nameByUser.get(row.key) : undefined) ||
        row.label ||
        "Someone",
      requests: row.requests,
    })),
  })
}
