import { canManage } from "@/features/organization/roles"
import { useOrganizationContext } from "@/shared/api/organizations"
import { useAliases, useRoutingPolicies } from "@/shared/api/routing"
import { useIsDeploymentScope } from "@/shared/hooks/DeploymentScope"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"

import { aliasAsRow, type RoutingRow } from "./policyModel"

/**
 * The policies and aliases this page lists, and what the caller may do with
 * them, for the list and for a policy's own page.
 *
 * In a workspace, the ones in force in the selected workspace: resolution is
 * workspace-keyed, so a stored policy belongs on its workspace's page, and
 * config-file entries have no workspace and are listed in every one. On the
 * deployment's own page, every policy on the deployment, unfiltered.
 */
export function useRoutingRows() {
  const organization = useOrganizationContext()
  const isDeploymentWide = useIsDeploymentScope()
  // Null only for a caller in no workspace, who has nowhere to write.
  const { selected } = useSelectedWorkspace()
  const workspaceId = isDeploymentWide ? undefined : selected?.workspace_id
  const policies = useRoutingPolicies(workspaceId)
  const aliases = useAliases(workspaceId)
  // Where a create lands. A write to an existing row uses the row's own
  // workspace instead, so a list refetched mid-switch cannot move a row.
  const writeWorkspaceId = selected?.workspace_id ?? null
  const canEdit =
    isDeploymentWide ||
    (canManage(organization.data) && writeWorkspaceId !== null)

  // Listed together: an alias is the one-target case of a policy.
  const rows: RoutingRow[] = [
    ...(policies.data ?? []).map((policy) => ({
      ...policy,
      kind: "policy" as const,
    })),
    ...(aliases.data ?? []).map(aliasAsRow),
  ].sort(
    (a, b) =>
      a.name.localeCompare(b.name) ||
      (a.user_id ?? "").localeCompare(b.user_id ?? ""),
  )

  return {
    rows,
    // Until the context settles neither list has been asked, and an empty
    // table would read as "no policies" rather than "not yet".
    isLoading:
      organization.isPending || policies.isLoading || aliases.isLoading,
    // A row missing from a list being refetched may be one just renamed into it.
    isFetching: policies.isFetching || aliases.isFetching,
    error: policies.error ?? aliases.error,
    isDeploymentWide,
    canEdit,
    writeWorkspaceId,
  }
}
