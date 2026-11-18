import { useState } from "react"
import {
  useIsOrganizationManager,
  useIsRoleSettled,
} from "@/features/organization/access"
import {
  buildMemberRules,
  buildRules,
  type Rule,
  type RuleSubject,
} from "@/features/rules/rules"
import { useKeys } from "@/shared/api/apiKeys"
import {
  useAllSpendCeilings,
  useOrganizationBudgets,
} from "@/shared/api/budgets"
import {
  useOrganizationContext,
  useOrganizationMembers,
} from "@/shared/api/organizations"
import { useOrganizationGuardrails } from "@/shared/api/tools"
import { useRefusedUsage } from "@/shared/api/usage"
import {
  useAllWorkspaceBudgetDefaults,
  useWorkspaces,
} from "@/shared/api/workspaces"

/**
 * Every rule the caller may read, asking only what the server answers them.
 * Ceilings, budgets, guardrails and the workspace list are owners' and
 * admins' reads; the roster, the caller's own keys and the per-person default
 * of each workspace they are in are anyone's.
 */
export function useRules(): {
  rules: Rule[]
  isManager: boolean
  isLoading: boolean
  error: unknown
} {
  const isSettled = useIsRoleSettled()
  const isManager = useIsOrganizationManager()
  const contextRead = useOrganizationContext()
  const context = contextRead.data
  // A period that ended before the page opened reads as nothing spent, as
  // Home reads it.
  const [now] = useState(() => Date.now())
  const ceilings = useAllSpendCeilings(isManager)
  const budgets = useOrganizationBudgets(isManager)
  const guardrails = useOrganizationGuardrails(isManager)
  const workspaces = useWorkspaces(isManager)
  const members = useOrganizationMembers(isSettled)
  const keys = useKeys(undefined, isSettled)
  const workspaceIds = !isSettled
    ? []
    : isManager
      ? (workspaces.data ?? []).map((one) => one.id)
      : (context?.workspace_memberships ?? []).map((one) => one.workspace_id)
  const defaults = useAllWorkspaceBudgetDefaults(workspaceIds)

  if (!isSettled) {
    return {
      rules: [],
      isManager,
      isLoading: !contextRead.error,
      error: contextRead.error,
    }
  }
  if (!isManager) {
    return {
      rules: buildMemberRules({
        members: members.data ?? [],
        keys: keys.data ?? [],
        defaults: defaults.data,
        callerMemberId: context?.organization_member_id,
        now,
      }),
      isManager,
      // The defaults name the rows, so the list waits for them rather than
      // renaming its rows as they land.
      isLoading: members.isLoading || keys.isLoading || defaults.isLoading,
      error: members.error ?? keys.error ?? defaults.error,
    }
  }
  return {
    rules: buildRules({
      ceilings: ceilings.data ?? [],
      budgets: budgets.data ?? [],
      members: members.data ?? [],
      keys: keys.data ?? [],
      workspaces: workspaces.data ?? [],
      guardrails: guardrails.data ?? [],
      defaults: defaults.data,
      organizationName: context?.organization?.name,
      now,
    }),
    isManager,
    // A per-person rule that is a workspace's default is changed through it,
    // so the list waits for the defaults as a member's does.
    isLoading:
      ceilings.isLoading ||
      budgets.isLoading ||
      guardrails.isLoading ||
      members.isLoading ||
      keys.isLoading ||
      workspaces.isLoading ||
      defaults.isLoading,
    error:
      ceilings.error ??
      budgets.error ??
      guardrails.error ??
      members.error ??
      keys.error ??
      workspaces.error ??
      defaults.error,
  }
}

/** The most refusals the list shows; the read asks one more to tell if there are others. */
export const REFUSALS_SHOWN = 50

/**
 * Requests the gateway refused with a 403 since `start`, for whoever a rule
 * reaches, or the whole organization for a rule on everyone. The usage log
 * does not say which rule refused a request, so these are every refusal its
 * subject met, newest first.
 */
export function useRefusedRequests(
  subject: RuleSubject,
  start: string,
  isEnabled: boolean,
) {
  return useRefusedUsage(subject, start, REFUSALS_SHOWN + 1, isEnabled)
}
