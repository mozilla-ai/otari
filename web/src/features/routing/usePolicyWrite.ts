import type { PolicySpec } from "@/client"
import {
  useCreateAlias,
  useCreateOrganizationAlias,
  useSetOrganizationRoutingPolicy,
  useSetRoutingPolicy,
} from "@/shared/api/routing"

import { findFallthroughTarget, type RoutingRow } from "./policyModel"

/**
 * Write a stored policy or alias back, on whichever of the four surfaces it
 * belongs to.
 *
 * An alias goes back through the alias API: it is still a row in
 * model_aliases, and rewriting it as a policy would leave the original behind
 * under the same name. Each caller gets its own mutations, so a refusal shows
 * beside the section that sent it.
 */
export function usePolicyWrite(
  row: RoutingRow,
  {
    isDeploymentWide,
    fallbackWorkspaceId,
  }: { isDeploymentWide: boolean; fallbackWorkspaceId: string | null },
) {
  const save = useSetRoutingPolicy()
  const saveAlias = useCreateAlias()
  const saveOrgPolicy = useSetOrganizationRoutingPolicy()
  const saveOrgAlias = useCreateOrganizationAlias()
  // The row's own workspace, not the selected one: an edit must not land a
  // second policy of this name somewhere else.
  const workspaceId = row.workspace_id ?? fallbackWorkspaceId
  const isTenantScoped = !isDeploymentWide && workspaceId !== null

  /** Save `spec`, under `name` when it differs, which renames the policy. */
  const write = (
    spec: PolicySpec,
    { name = row.name, onSuccess }: { name?: string; onSuccess?: () => void },
  ) => {
    const options = { onSuccess: () => onSuccess?.() }
    if (row.kind === "alias") {
      const target = findFallthroughTarget(spec)
      if (isTenantScoped)
        saveOrgAlias.mutate(
          { name: row.name, target, workspace_id: workspaceId },
          options,
        )
      else
        saveAlias.mutate(
          {
            name: row.name,
            target,
            user_id: row.user_id ?? null,
            workspace_id: workspaceId,
          },
          options,
        )
      return
    }
    // A name is the key, so a rename travels as `rename_from` on the same
    // write rather than as a second policy.
    const rename = name !== row.name ? { rename_from: row.name } : {}
    if (isTenantScoped)
      saveOrgPolicy.mutate(
        { name, spec, workspace_id: workspaceId, ...rename },
        options,
      )
    else
      save.mutate(
        {
          name,
          spec,
          user_id: row.user_id ?? null,
          workspace_id: workspaceId,
          ...rename,
        },
        options,
      )
  }

  return {
    write,
    isPending:
      save.isPending ||
      saveAlias.isPending ||
      saveOrgPolicy.isPending ||
      saveOrgAlias.isPending,
    error:
      save.error ??
      saveAlias.error ??
      saveOrgPolicy.error ??
      saveOrgAlias.error,
  }
}
