/**
 * What the caller's roles let them do, for hiding controls the server would
 * refuse. The server still decides; these only keep a refusal from being
 * offered. Read as allowed until the context answers, so a control does not
 * blink in for everyone who may use it.
 */

import {
  canContributeToWorkspace,
  canManage,
  canManageWorkspace,
} from "@/features/organization/roles"
import {
  useDeploymentOperator,
  useOrganizationContext,
} from "@/shared/api/organizations"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"

/**
 * Whether the caller may mint a key in the selected workspace. A key belongs
 * to a workspace, so someone placed in none has nowhere to mint one.
 */
export function useCanCreateKey(): boolean {
  const context = useOrganizationContext().data
  const { selected, isSettled } = useSelectedWorkspace()
  const { answer, isOperator } = useDeploymentOperator()
  // No organization is an answer, not a wait: there is no workspace to mint in.
  if (answer === "no-organization") return false
  if (context === undefined || !isSettled) return true
  if (isOperator) return true
  return selected !== null && canContributeToWorkspace(context, selected.role)
}

/**
 * Whether the caller is an owner or admin of the organization, false until
 * the context has answered: for reads the server refuses anyone else, asking
 * early is a refused request. Deployment authority is not asked: the server
 * ignores it for tenant pages.
 */
export function useIsOrganizationManager(): boolean {
  const context = useOrganizationContext().data
  return context !== undefined && canManage(context)
}

/**
 * Which workspaces the caller may manage: every one for an organization owner
 * or admin, and otherwise each one they are an owner or admin of themselves,
 * which the server allows an organization member (its members, per-person
 * default, provider policy, MCP servers, trace content). Strict, like
 * `useIsOrganizationManager`: nothing until the context has answered.
 */
export function useCanManageWorkspace(): (workspaceId: string) => boolean {
  const context = useOrganizationContext().data
  return (workspaceId) =>
    canManageWorkspace(
      context,
      context?.workspace_memberships?.find(
        (one) => one.workspace_id === workspaceId,
      )?.role,
    )
}

/** Whether the caller's roles are known yet. */
export function useIsRoleSettled(): boolean {
  return useOrganizationContext().data !== undefined
}
