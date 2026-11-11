import { Link } from "@tanstack/react-router"

import { ControlField } from "@/design-system/forms/FieldMessages"
import { canManageWorkspace } from "@/features/organization/roles"
import { decidedByLabel, keyAccessLabel } from "@/features/tools/mcpAccess"
import { useOrganizationContext } from "@/shared/api/organizations"
import { useWorkspaceMcpKeyAccess } from "@/shared/api/tools"

/**
 * What one key reaches on each of its workspace's MCP servers, read-only, for
 * the key editor. Changing it happens on each server's page, which every row
 * links to, because the access list belongs to the server rather than the key.
 *
 * Shown only to someone who manages the key's workspace, and not at all when
 * the workspace has no servers or the read is refused: a key form has nothing
 * to say about MCP then.
 */
export function KeyMcpAccess({
  workspaceId,
  apiKeyId,
}: {
  workspaceId: string
  apiKeyId: string
}) {
  const context = useOrganizationContext()
  const role = context.data?.workspace_memberships?.find(
    (membership) => membership.workspace_id === workspaceId,
  )?.role
  const canManage = canManageWorkspace(context.data, role)
  const access = useWorkspaceMcpKeyAccess(workspaceId, apiKeyId, canManage)
  const servers = access.data ?? []
  if (!canManage || servers.length === 0) return null

  return (
    <section aria-label="MCP access" className="flex flex-col gap-2">
      <ControlField
        label="MCP access"
        description="Set in each server's ACL. A rule for the key wins over a rule for its owner, and either wins over the server's default policy."
      />
      <ul className="flex flex-col gap-2">
        {servers.map((server) => (
          <li
            key={server.server_id}
            className="flex min-w-0 flex-col gap-0.5 sm:flex-row sm:items-baseline sm:gap-2"
          >
            <Link
              to="/tools/mcp-servers/$serverId"
              params={{ serverId: server.server_id }}
              className="shrink-0 text-link hover:text-link-hover"
            >
              {server.name}
            </Link>
            <span className="min-w-0 break-words text-body">
              {server.enabled ? keyAccessLabel(server) : "Server disabled"}
            </span>
            <span className="text-caption">
              matched {decidedByLabel(server.decided_by)}
            </span>
          </li>
        ))}
      </ul>
    </section>
  )
}
